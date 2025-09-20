import grpc
from concurrent import futures
import psycopg2
from psycopg2 import pool
from psycopg2.errors import UniqueViolation
import bcrypt
import logging
import secrets
import random
import jwt
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from functools import wraps

# --- Load Environment Variables ---
# This will find the .env file in the root project folder
load_dotenv("credentials.env")

# --- Local, Relative Imports ---
from . import auth_pb2
from . import auth_pb2_grpc
from .email_utils import send_otp_email, send_password_reset_email, send_admin_approval_email

# --- CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Security & App Settings from Environment ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default-insecure-secret-key")
JWT_ALGORITHM = "HS256"
OTP_EXPIRATION_MINUTES = 10
PASSWORD_RESET_EXPIRATION_HOURS = 1
APP_BASE_URL = "http://localhost:8501"

# --- Database Connection Pool ---
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1, maxconn=10,
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost")
    )
    logging.info("Database connection pool created successfully.")
except (psycopg2.OperationalError, TypeError) as e:
    logging.critical(f"FATAL: Could not connect to the database. Check your .env file. Error: {e}")
    db_pool = None

# --- HELPER FUNCTIONS ---
def _generate_otp():
    return str(random.randint(100000, 999999))

def _get_utc_now():
    return datetime.now(timezone.utc)

def _hash_value(value: str) -> str:
    return bcrypt.hashpw(value.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def _check_hash(value: str, hashed_value: str) -> bool:
    if value and hashed_value:
        return bcrypt.checkpw(value.encode("utf-8"), hashed_value.encode("utf-8"))
    return False

# NEW: Decorator to handle database connection and transactions
def db_connection(func):
    @wraps(func)
    def wrapper(self, request, context):
        conn = None
        try:
            if not db_pool:
                raise ConnectionError("Database pool is unavailable.")
            conn = db_pool.getconn()
            # Pass the connection and cursor to the function
            with conn.cursor() as cursor:
                result = func(self, request, context, conn, cursor)
            conn.commit()
            return result
        except (UniqueViolation, ValueError) as e:
            if conn: conn.rollback()
            # Re-raise specific exceptions to be handled by the calling method
            raise e
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Database error in {func.__name__}: {e}")
            # For general errors, set a gRPC status
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("An internal database error occurred.")
            # Return a default/empty response object if possible
            # This part might need adjustment based on return types
            return func.__annotations__.get('return')()
        finally:
            if db_pool and conn:
                db_pool.putconn(conn)
    return wrapper


# --- GRPC SERVICE IMPLEMENTATION ---
class AuthService(auth_pb2_grpc.AuthServiceServicer):

    @db_connection
    def RegisterUser(self, request, context, conn, cursor):
        try:
            hashed_pw = _hash_value(request.password)
            otp = _generate_otp()
            hashed_otp = _hash_value(otp)
            expiry = _get_utc_now() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
            
            cursor.execute(
                """
                INSERT INTO users (username, email, hashed_password, phone_number, role, status, verification_otp_hash, verification_otp_expiry)
                VALUES (%s, %s, %s, %s, 'user', 'pending_verification', %s, %s)
                """,
                (request.username, request.email, hashed_pw, request.phone_number, hashed_otp, expiry),
            )
            
            send_otp_email(request.email, request.username, otp)
            logging.info(f"Registration for {request.email} successful. OTP sent.")
            return auth_pb2.AuthResponse(success=True, message="Registration successful! Check your email for an OTP.")
        except UniqueViolation:
            return auth_pb2.AuthResponse(success=False, message="Username or email already exists.")

    # IMPLEMENTED: New function
    @db_connection
    def RegisterAdmin(self, request, context, conn, cursor):
        try:
            hashed_pw = _hash_value(request.password)
            cursor.execute(
                """
                INSERT INTO users (username, email, hashed_password, full_name, employee_id, admin_request_reason, role, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'admin', 'pending_approval')
                """,
                (request.username, request.email, hashed_pw, request.full_name, request.employee_id, request.admin_request_reason),
            )
            logging.info(f"Admin registration request for {request.email} submitted.")
            return auth_pb2.AuthResponse(success=True, message="Admin request submitted successfully.")
        except UniqueViolation:
            return auth_pb2.AuthResponse(success=False, message="Username or email already exists.")


    @db_connection
    def VerifyAccountWithOTP(self, request, context, conn, cursor):
        cursor.execute("SELECT id, verification_otp_hash, verification_otp_expiry FROM users WHERE email = %s AND status = 'pending_verification'", (request.email,))
        user = cursor.fetchone()
        if not user:
            return auth_pb2.AuthResponse(success=False, message="User not found or already verified.")
        
        user_id, stored_otp_hash, expiry_time = user
        if _get_utc_now() > expiry_time:
            return auth_pb2.AuthResponse(success=False, message="OTP has expired.")
        if not _check_hash(request.otp_code, stored_otp_hash):
            return auth_pb2.AuthResponse(success=False, message="Invalid OTP.")

        cursor.execute("UPDATE users SET status = 'active', verification_otp_hash = NULL, verification_otp_expiry = NULL WHERE id = %s", (user_id,))
        logging.info(f"Account for {request.email} verified.")
        return auth_pb2.AuthResponse(success=True, message="Account verified successfully! You can now log in.")

    # MODIFIED: Now includes Login Logging
    def Login(self, request, context):
        conn = None
        user_id = None
        is_success = False
        try:
            conn = db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, hashed_password, role, status FROM users WHERE email = %s", (request.email,))
                user = cursor.fetchone()

                if not user:
                    return auth_pb2.LoginResponse(success=False, message="Invalid email or password.")
                
                user_id, hashed_password, role, status = user
                
                if status != 'active':
                    return auth_pb2.LoginResponse(success=False, message=f"Account not active. Status: {status}.")
                
                if not _check_hash(request.password, hashed_password):
                    return auth_pb2.LoginResponse(success=False, message="Invalid email or password.")
                
                is_success = True
                payload = {'user_id': str(user_id), 'role': role, 'exp': _get_utc_now() + timedelta(hours=24)}
                token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

                logging.info(f"User {request.email} logged in.")
                return auth_pb2.LoginResponse(success=True, message="Login successful.", token=token, user_id=str(user_id), role=role)
        
        except Exception as e:
            logging.error(f"Error in Login: {e}")
            return auth_pb2.LoginResponse(success=False, message="An internal error occurred.")
        finally:
            # Log the login attempt
            try:
                if conn:
                    with conn.cursor() as log_cursor:
                        peer = context.peer() # Get client IP
                        log_cursor.execute(
                            """
                            INSERT INTO login_logs (user_id, email_attempt, ip_address, is_success)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (user_id, request.email, peer, is_success)
                        )
                    conn.commit()
            except Exception as log_e:
                logging.error(f"Failed to write to login_logs: {log_e}")
                if conn: conn.rollback()
            
            if db_pool and conn:
                db_pool.putconn(conn)


    @db_connection
    def GetPendingAdmins(self, request, context, conn, cursor):
        cursor.execute(
            """
            SELECT id, username, email, full_name, admin_request_reason, created_at 
            FROM users WHERE role = 'admin' AND status = 'pending_approval' ORDER BY created_at ASC
            """
        )
        pending_admins = [
            auth_pb2.AdminUser(
                user_id=str(row[0]), username=row[1], email=row[2],
                full_name=row[3], admin_request_reason=row[4] or "",
                created_at=row[5].isoformat(),
            ) for row in cursor.fetchall()
        ]
        return auth_pb2.AdminListResponse(admins=pending_admins)

    @db_connection
    def ApproveAdmin(self, request, context, conn, cursor):
        cursor.execute("SELECT email FROM users WHERE id = %s", (request.user_id,))
        user_record = cursor.fetchone()
        user_to_approve_email = user_record[0] if user_record else None

        cursor.execute(
            "UPDATE users SET status = 'active' WHERE id = %s AND role = 'admin' AND status = 'pending_approval'",
            (request.user_id,)
        )
        if cursor.rowcount == 0:
            return auth_pb2.AuthResponse(success=False, message="Admin not found or already approved.")
        
        if user_to_approve_email:
            send_admin_approval_email(user_to_approve_email)
            logging.info(f"Admin {request.user_id} approved and notified.")
        return auth_pb2.AuthResponse(success=True, message="Admin approved successfully.")
    
    # IMPLEMENTED: New function
    @db_connection
    def CheckUsername(self, request, context, conn, cursor):
        cursor.execute("SELECT id FROM users WHERE username = %s", (request.username,))
        is_taken = cursor.fetchone() is not None
        return auth_pb2.CheckUsernameResponse(is_available=not is_taken)

    # IMPLEMENTED: New functions
    def _update_user_status(self, user_id, new_status, allowed_old_status):
        conn = None
        try:
            conn = db_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE users SET status = %s WHERE id = %s AND status = %s",
                    (new_status, user_id, allowed_old_status)
                )
                if cursor.rowcount == 0:
                    return auth_pb2.AuthResponse(success=False, message="User not found or status condition not met.")
                conn.commit()
            logging.info(f"User {user_id} status updated to {new_status}.")
            return auth_pb2.AuthResponse(success=True, message=f"User status updated to {new_status}.")
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error updating status for {user_id}: {e}")
            return auth_pb2.AuthResponse(success=False, message="An internal error occurred.")
        finally:
            if db_pool and conn:
                db_pool.putconn(conn)
    
    def SuspendUser(self, request, context):
        logging.info(f"Attempting to suspend user {request.user_id}. Reason: {request.reason}")
        return self._update_user_status(request.user_id, 'suspended', 'active')

    def UnsuspendUser(self, request, context):
        logging.info(f"Attempting to unsuspend user {request.user_id}.")
        return self._update_user_status(request.user_id, 'active', 'suspended')


# --- SERVER STARTUP ---
def serve():
    if not db_pool:
        logging.critical("Cannot start server: Database connection pool is unavailable.")
        return
        
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("gRPC Auth Service started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()