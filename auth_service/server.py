import grpc
from concurrent import futures
import psycopg2
from psycopg2 import pool
from psycopg2.errors import UniqueViolation
import bcrypt
import logging
import random
import jwt
import os
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import urllib.parse

# Google ID token verification library
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

# --- Load Environment Variables ---
load_dotenv("credentials.env")

# --- Local, Relative Imports ---
from . import auth_pb2
from . import auth_pb2_grpc
from .email_utils import send_otp_email, send_password_reset_email, send_admin_approval_email

# --- CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Security & App Settings from Environment ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "a-very-insecure-default-secret-key")
JWT_ALGORITHM = "HS256"
OTP_EXPIRATION_MINUTES = 10
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

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
    logging.critical(f"FATAL: Could not connect to the database. Check your credentials.env file. Error: {e}")
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

def _verify_google_id_token(id_tok: str):
    if not GOOGLE_CLIENT_ID:
        raise ValueError("GOOGLE_CLIENT_ID is not configured on the server.")
    req = google_requests.Request()
    id_info = google_id_token.verify_oauth2_token(id_tok, req, GOOGLE_CLIENT_ID)
    return id_info

# --- GRPC SERVICE IMPLEMENTATION ---
class AuthServiceServicer(auth_pb2_grpc.AuthServiceServicer):
    
    def _get_db_conn(self):
        """Gets a connection from the pool."""
        return db_pool.getconn()

    def _release_db_conn(self, conn):
        """Releases a connection back to the pool."""
        if conn:
            db_pool.putconn(conn)

    def RegisterAdmin(self, request, context):
        """
        Handles a new user request for admin access.
        Saves the user to the DB with role='admin' and status='pending_approval'.
        """
        conn = None
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cur:
                hashed_password = _hash_value(request.password)
                user_id_variable = str(uuid.uuid4())
                
                cur.execute(
                    """
                    INSERT INTO users (id, full_name, email, username, hashed_password, role, status, employee_id, admin_request_reason)
                    VALUES (%s, %s, %s, %s, %s, 'admin', 'pending_approval', %s, %s)
                    """,
                    (
                        user_id_variable,
                        request.full_name,
                        request.email,
                        request.username,
                        hashed_password,
                        request.employee_id,
                        request.admin_request_reason 
                    )
                )
                conn.commit()
                
                logging.info(f"Admin request for user '{request.username}' submitted for approval.")
                return auth_pb2.AuthResponse(
                    success=True,
                    message="Admin request submitted successfully. Please wait for an administrator to approve your account."
                )

        except UniqueViolation:
            if conn: conn.rollback()
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            context.set_details("Username or email already exists.")
            return auth_pb2.AuthResponse(success=False, message="Username or email already exists.")
            
        except Exception as e:
            if conn: conn.rollback()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            logging.error(f"Error in RegisterAdmin: {e}")
            return auth_pb2.AuthResponse(success=False, message="An internal server error occurred.")

        finally:
            self._release_db_conn(conn)

    def RequestAdminAccess(self, request, context):
        """Handles a new user request for admin access."""
        return self.RegisterAdmin(request, context)
    
    def _log_login_attempt(self, conn, user_id, email, peer_string, user_agent, is_success):
        """Helper to parse peer string and insert a record into the login_logs table."""
        try:
            ip_address = "Unknown"
            if peer_string:
                unquoted = urllib.parse.unquote(peer_string)
                ip_part = unquoted.split(":", 1)[1] if ":" in unquoted else unquoted
                ip_address = ip_part.rsplit(':', 1)[0].strip("[]")

            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO login_logs (user_id, email_attempt, ip_address, user_agent, is_success)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, email, ip_address, user_agent, is_success)
                )
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to write to login_logs: {e}")
            if conn: conn.rollback()

    def RegisterUser(self, request, context):
        """Handles new user registration with local credentials."""
        conn = None
        try:
            conn = self._get_db_conn()
            hashed_pw = _hash_value(request.password)
            otp = _generate_otp()
            hashed_otp = _hash_value(otp)
            expiry = _get_utc_now() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
            
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (username, email, hashed_password, phone_number, role, status, verification_otp_hash, verification_otp_expiry)
                    VALUES (%s, %s, %s, %s, 'user', 'pending_verification', %s, %s)
                    """,
                    (request.username, request.email, hashed_pw, request.phone_number, hashed_otp, expiry),
                )
                conn.commit()
            
            send_otp_email(request.email, request.username, otp)
            return auth_pb2.AuthResponse(success=True, message="Registration successful! Check your email for an OTP.")
        except UniqueViolation:
            if conn: conn.rollback()
            return auth_pb2.AuthResponse(success=False, message="Username or email already exists.")
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error in RegisterUser: {e}")
            return auth_pb2.AuthResponse(success=False, message="An internal server error occurred.")
        finally:
            self._release_db_conn(conn)

    def VerifyAccountWithOTP(self, request, context):
        """Verifies a user's account with the provided OTP."""
        conn = None
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, verification_otp_hash, verification_otp_expiry FROM users WHERE email = %s AND status = 'pending_verification'",
                    (request.email,)
                )
                user = cursor.fetchone()
                if not user:
                    return auth_pb2.AuthResponse(success=False, message="Invalid email or account already verified.")

                user_id, hashed_otp, expiry = user
                if not hashed_otp or not expiry or _get_utc_now() > expiry:
                    return auth_pb2.AuthResponse(success=False, message="OTP has expired or is invalid.")
                
                if not _check_hash(request.otp_code, hashed_otp):
                    return auth_pb2.AuthResponse(success=False, message="Invalid OTP.")

                cursor.execute(
                    "UPDATE users SET status = 'active', verification_otp_hash = NULL, verification_otp_expiry = NULL WHERE id = %s",
                    (user_id,)
                )
                conn.commit()
            return auth_pb2.AuthResponse(success=True, message="Account verified successfully. You can now log in.")
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error in VerifyAccountWithOTP: {e}")
            return auth_pb2.AuthResponse(success=False, message="An internal server error occurred.")
        finally:
            self._release_db_conn(conn)
            
    def Login(self, request, context):
        """Handles traditional email/password login and returns user info."""
        conn = None
        user_id = None
        is_success = False
        user_agent = dict(context.invocation_metadata()).get('user-agent', 'Unknown')
        
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, username, email, hashed_password, role, status FROM users WHERE email = %s AND auth_provider = 'local'",
                    (request.email,)
                )
                user = cursor.fetchone()
                if not user:
                    return auth_pb2.LoginResponse(success=False, message="Invalid email or password.")
                
                user_id, username, email, hashed_password, role, status = user
                
                if status != 'active':
                    return auth_pb2.LoginResponse(success=False, message=f"Account not active. Status: {status}.")
                if not _check_hash(request.password, hashed_password):
                    return auth_pb2.LoginResponse(success=False, message="Invalid email or password.")
                
                is_success = True
                role_enum = auth_pb2.UserRole.Value(role.upper())
                payload = {'user_id': str(user_id), 'role': role, 'exp': _get_utc_now() + timedelta(hours=24)}
                token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
                
                logging.info(f"User {email} logged in successfully.")
                return auth_pb2.LoginResponse(
                    success=True, message="Login successful.", token=token, 
                    user_id=str(user_id), role=role_enum, username=username, email=email
                )
        except Exception as e:
            logging.error(f"Error in Login: {e}")
            return auth_pb2.LoginResponse(success=False, message="An internal server error occurred.")
        finally:
            if conn:
                self._log_login_attempt(conn, user_id, request.email, context.peer(), user_agent, is_success)
                self._release_db_conn(conn)

    def GetPendingAdmins(self, request, context):
        """Fetches a list of admin accounts that are pending approval."""
        conn = None
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, username, email, full_name, admin_request_reason, created_at 
                    FROM users WHERE role = 'admin' AND status = 'pending_approval' ORDER BY created_at ASC
                    """
                )
                pending_admins = []
                for row in cursor.fetchall():
                    pending_admins.append(auth_pb2.AdminUser(
                        user_id=str(row[0]),
                        username=row[1],
                        email=row[2],
                        full_name=row[3] or "",
                        admin_request_reason=row[4] or "",
                        created_at=row[5].isoformat()
                    ))
                return auth_pb2.AdminListResponse(admins=pending_admins)
        except Exception as e:
            logging.error(f"Error in GetPendingAdmins: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to fetch pending admins.")
            return auth_pb2.AdminListResponse()
        finally:
            self._release_db_conn(conn)

    # NEW METHOD: Handles admin approval, role expiry, and email notification.
    def ApproveAdmin(self, request, context):
        """
        Approves a pending admin, sets their status to 'active',
        optionally sets a role expiry date, and sends a notification email.
        """
        conn = None
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cur:
                expiry_date = None
                expiry_date_str = ""
                # Calculate the expiry date if a positive number of days is given
                if request.expiry_days > 0:
                    expiry_date = datetime.now(timezone.utc) + timedelta(days=request.expiry_days)
                    expiry_date_str = expiry_date.strftime("%Y-%m-%d")

                # Update user status and optional role expiry date
                # Also fetch the user's details for the email
                cur.execute(
                    """
                    UPDATE users
                    SET status = 'active', role_expiry_at = %s
                    WHERE id = %s AND status = 'pending_approval'
                    RETURNING email, full_name;
                    """,
                    (expiry_date, request.user_id)
                )
                
                updated_user = cur.fetchone()
                
                if not updated_user:
                    conn.rollback()
                    return auth_pb2.AuthResponse(success=False, message="User not found or already approved.")

                conn.commit()
                
                # Send the approval email
                user_email, user_full_name = updated_user
                send_admin_approval_email(user_email, user_full_name, expiry_date_str)

                return auth_pb2.AuthResponse(success=True, message="Admin approved successfully.")

        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error in ApproveAdmin: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("An internal server error occurred.")
            return auth_pb2.AuthResponse(success=False, message="An internal server error occurred.")
        finally:
            self._release_db_conn(conn)
            
    def LoginWithOAuth(self, request, context):
        """Handles OAuth2 login, creates/updates user, and returns user info."""
        conn = None
        user_id = None
        email_from_token = ""
        is_success = False
        user_agent = dict(context.invocation_metadata()).get('user-agent', 'Unknown')

        try:
            if request.provider.upper() != "GOOGLE":
                return auth_pb2.LoginResponse(success=False, message="Unsupported OAuth provider.")
            
            id_info = _verify_google_id_token(request.id_token)
            provider_id = id_info.get('sub')
            email_from_token = id_info.get('email')
            full_name = id_info.get('name')
            avatar_url = id_info.get('picture')

            if not provider_id or not email_from_token:
                return auth_pb2.LoginResponse(success=False, message="Invalid OAuth token: missing required claims.")

            conn = self._get_db_conn()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (username, email, full_name, auth_provider, provider_id, avatar_url, role, status)
                    VALUES (%s, %s, %s, 'google', %s, %s, 'user', 'active')
                    ON CONFLICT (auth_provider, provider_id) DO UPDATE
                      SET email = EXCLUDED.email, full_name = EXCLUDED.full_name,
                          avatar_url = EXCLUDED.avatar_url, updated_at = NOW()
                    RETURNING id, role, username, email;
                    """,
                    (f"google_{provider_id[:12]}", email_from_token, full_name, provider_id, avatar_url)
                )
                user_id, role, db_username, db_email = cursor.fetchone()
                conn.commit()
            
            is_success = True
            role_enum = auth_pb2.UserRole.Value(role.upper())
            payload = {'user_id': str(user_id), 'role': role, 'exp': _get_utc_now() + timedelta(hours=24)}
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

            logging.info(f"OAuth user {db_email} logged in successfully.")
            return auth_pb2.LoginResponse(
                success=True, message="OAuth login successful.",
                token=token, user_id=str(user_id),
                role=role_enum, username=db_username, email=db_email
            )
        except ValueError as e: 
            return auth_pb2.LoginResponse(success=False, message=str(e))
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error during OAuth login: {e}")
            return auth_pb2.LoginResponse(success=False, message="An internal error occurred during OAuth login.")
        finally:
            if conn:
                self._log_login_attempt(conn, user_id, email_from_token, context.peer(), user_agent, is_success)
                self._release_db_conn(conn)

    def ForgotPassword(self, request, context):
        """Initiates the password reset process by sending an OTP."""
        conn = None
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE email = %s AND auth_provider = 'local' AND status = 'active'", (request.email,))
                user = cursor.fetchone()
                if user:
                    user_id = user[0]
                    otp = _generate_otp()
                    hashed_otp = _hash_value(otp)
                    expiry = _get_utc_now() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
                    
                    cursor.execute(
                        "UPDATE users SET reset_password_otp_hash = %s, reset_password_otp_expiry = %s WHERE id = %s",
                        (hashed_otp, expiry, user_id)
                    )
                    conn.commit()
                    send_password_reset_email(request.email, otp)
            
            return auth_pb2.AuthResponse(success=True, message="If an account with that email exists, a password reset OTP has been sent.")
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error in ForgotPassword: {e}")
            return auth_pb2.AuthResponse(success=False, message="An internal server error occurred.")
        finally:
            self._release_db_conn(conn)

    def ResetPassword(self, request, context):
        """Resets the user's password using a valid OTP."""
        conn = None
        try:
            conn = self._get_db_conn()
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, reset_password_otp_hash, reset_password_otp_expiry FROM users WHERE email = %s AND status = 'active'",
                    (request.email,)
                )
                user = cursor.fetchone()
                if not user:
                    return auth_pb2.AuthResponse(success=False, message="Invalid email or OTP.")

                user_id, hashed_otp, expiry = user
                if not hashed_otp or not expiry or _get_utc_now() > expiry:
                    return auth_pb2.AuthResponse(success=False, message="OTP has expired or is invalid.")
                
                if not _check_hash(request.token, hashed_otp):
                    return auth_pb2.AuthResponse(success=False, message="Invalid email or OTP.")

                new_hashed_password = _hash_value(request.new_password)
                cursor.execute(
                    "UPDATE users SET hashed_password = %s, reset_password_otp_hash = NULL, reset_password_otp_expiry = NULL WHERE id = %s",
                    (new_hashed_password, user_id)
                )
                conn.commit()

            return auth_pb2.AuthResponse(success=True, message="Password has been reset successfully.")
        except Exception as e:
            if conn: conn.rollback()
            logging.error(f"Error in ResetPassword: {e}")
            return auth_pb2.AuthResponse(success=False, message="An internal server error occurred.")
        finally:
            self._release_db_conn(conn)
    
# --- SERVER STARTUP ---
def serve():
    """Initializes and starts the gRPC server."""
    if not db_pool:
        logging.critical("Cannot start server: Database connection pool is unavailable.")
        return
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    auth_pb2_grpc.add_AuthServiceServicer_to_server(AuthServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("gRPC Auth Service started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()