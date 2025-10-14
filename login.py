import streamlit as st
import grpc
import re  # NEW: Import for regular expression validation
from streamlit_oauth import OAuth2Component
from st_phonenumbers_input import st_phonenumbers_input  # NEW: Import for phone number input

# --- Local Imports ---
from auth_service import auth_pb2, auth_pb2_grpc

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Authentication | ML App", layout="centered")
st.markdown("<style>[data-testid='stSidebar'] {display: none;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# --- GRPC & OAUTH SETUP ---
@st.cache_resource
def get_grpc_stub():
    """Creates and caches the gRPC stub for server communication."""
    try:
        channel = grpc.insecure_channel("localhost:50051")
        grpc.channel_ready_future(channel).result(timeout=5)
        return auth_pb2_grpc.AuthServiceStub(channel)
    except grpc.FutureTimeoutError:
        st.error("Connection to the authentication server timed out. Please ensure the gRPC server is running.")
        return None
    except Exception as e:
        st.error(f"A gRPC connection error occurred: {e}")
        return None

stub = get_grpc_stub()
if not stub:
    st.stop()

# Load OAuth credentials
try:
    GOOGLE_CLIENT_ID = st.secrets["google_oauth"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["google_oauth"]["client_secret"]
    REDIRECT_URI = st.secrets["google_oauth"]["redirect_uri"]
except KeyError:
    st.error("Google OAuth credentials are not configured in .streamlit/secrets.toml.")
    st.stop()

# --- NEW: PASSWORD VALIDATION HELPER ---
def is_password_strong(password):
    """
    Checks if a password meets strength requirements.
    Returns (True, "OK") if strong, otherwise (False, "Error Message").
    """
    if len(password) < 8:
        return (False, "Password must be at least 8 characters long.")
    if not re.search(r"[a-z]", password):
        return (False, "Password must contain at least one lowercase letter.")
    if not re.search(r"[A-Z]", password):
        return (False, "Password must contain at least one uppercase letter.")
    if not re.search(r"\d", password):
        return (False, "Password must contain at least one number.")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return (False, "Password must contain at least one special character.")
    return (True, "OK")

# --- SESSION STATE INITIALIZATION ---
if 'authenticated' not in st.session_state:
    st.session_state.update({
        "authenticated": False, "token": None, "user_id": None, "role": None,
        "username": None, "email": None,
        "auth_page": "login", "pending_registration_email": None
    })

# --- REDIRECTION LOGIC ---
if st.session_state.get("authenticated"):
    role_name = auth_pb2.UserRole.Name(st.session_state.role)
    if role_name == 'ADMIN':
        st.switch_page("pages/2_🔑_Admin_Dashboard.py")
    else:
        st.switch_page("pages/3_🔮_Inference.py")

# --- HANDLER FUNCTIONS ---
def set_auth_state(response):
    """A helper function to update session state after any successful login."""
    st.session_state.authenticated = True
    st.session_state.token = response.token
    st.session_state.user_id = response.user_id
    st.session_state.role = response.role
    st.session_state.username = response.username
    st.session_state.email = response.email
    st.rerun()

def handle_login():
    """Handles traditional email/password login."""
    email = st.session_state.get("login_email", "")
    password = st.session_state.get("login_password", "")
    if not all([email, password]):
        st.warning("Please enter both email and password.")
        return
    try:
        req = auth_pb2.LoginRequest(email=email, password=password)
        metadata = [('user-agent', 'StreamlitFrontend/1.0')]
        response = stub.Login(req, metadata=metadata)
        if response.success:
            set_auth_state(response)
        else:
            st.error(f"Login Failed: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.details()}")

def handle_oauth_login(token):
    """Callback for successful OAuth, calls the gRPC backend."""
    if token and "id_token" in token:
        try:
            req = auth_pb2.LoginWithOAuthRequest(provider=auth_pb2.GOOGLE, id_token=token["id_token"])
            metadata = [('user-agent', 'StreamlitFrontend/1.0')]
            response = stub.LoginWithOAuth(req, metadata=metadata)
            if response.success:
                set_auth_state(response)
            else:
                st.error(f"OAuth Login Failed: {response.message}")
        except grpc.RpcError as e:
            st.error(f"Server communication error: {e.details()}")

def handle_user_register():
    email = st.session_state.get("user_reg_email", "")
    username = st.session_state.get("user_reg_username", "")
    phone = st.session_state.get("user_reg_phone") # The component returns a formatted string or None
    password = st.session_state.get("user_reg_password", "")
    if not all([email, username, password]):
        st.warning("Please fill all required fields.")
        return

    # MODIFIED: Added password strength validation
    is_strong, message = is_password_strong(password)
    if not is_strong:
        st.warning(message)
        return

    try:
        # Use the formatted phone number, or an empty string if it's None or invalid
        phone_number_to_send = phone if phone else ""
        req = auth_pb2.RegisterUserRequest(email=email, username=username, password=password, phone_number=phone_number_to_send)
        response = stub.RegisterUser(req)
        if response.success:
            st.session_state.pending_registration_email = email
            st.session_state.auth_page = "verify_otp"
            st.rerun()
        else:
            st.error(f"Registration Failed: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.details()}")

def handle_admin_register():
    email = st.session_state.get("admin_reg_email", "")
    username = st.session_state.get("admin_reg_username", "")
    password = st.session_state.get("admin_reg_password", "")
    full_name = st.session_state.get("admin_reg_fullname", "")
    employee_id = st.session_state.get("admin_reg_empid", "")
    reason = st.session_state.get("admin_reg_reason", "")
    if not all([email, username, password, full_name, reason]):
        st.warning("Please fill all required fields for admin registration.")
        return
        
    # MODIFIED: Added password strength validation
    is_strong, message = is_password_strong(password)
    if not is_strong:
        st.warning(message)
        return
        
    try:
        req = auth_pb2.RegisterAdminRequest(
            email=email, username=username, password=password,
            full_name=full_name, employee_id=employee_id, admin_request_reason=reason
        )
        response = stub.RegisterAdmin(req)
        if response.success:
            st.success("Admin request submitted successfully! It will be reviewed by an existing administrator.")
            st.session_state.auth_page = "login"
            st.rerun()
        else:
            st.error(f"Admin Registration Failed: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.details()}")

def handle_otp_verification():
    email = st.session_state.get("pending_registration_email")
    otp = st.session_state.get("otp_code", "")
    if not otp or not otp.isdigit() or len(otp) != 6:
        st.error("Please enter a valid 6-digit OTP.")
        return
    try:
        response = stub.VerifyAccountWithOTP(auth_pb2.VerifyOTPRequest(email=email, otp_code=otp))
        if response.success:
            st.success("Verification successful! You can now log in.")
            st.session_state.auth_page = "login"
            st.session_state.pending_registration_email = None
            st.rerun()
        else:
            st.error(f"Verification Failed: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.details()}")

def handle_forgot_password():
    email = st.session_state.get("forgot_email", "")
    if not email:
        st.warning("Please enter your email address.")
        return
    try:
        response = stub.ForgotPassword(auth_pb2.ForgotPasswordRequest(email=email))
        if response.success:
            st.success(response.message)
            st.info("Please check your email for the OTP and proceed to reset your password.")
            st.session_state.auth_page = "reset_password"
            st.session_state.reset_email = email
            st.rerun()
        else:
            st.error(f"Error: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.details()}")

def handle_reset_password():
    email = st.session_state.get("reset_email")
    otp = st.session_state.get("reset_otp", "")
    new_password = st.session_state.get("reset_new_password", "")
    confirm_password = st.session_state.get("reset_confirm_password", "")
    if not all([otp, new_password, confirm_password]):
        st.error("Please fill out all fields.")
        return
    if new_password != confirm_password:
        st.error("Passwords do not match.")
        return
        
    # MODIFIED: Added password strength validation for reset password
    is_strong, message = is_password_strong(new_password)
    if not is_strong:
        st.warning(message)
        return
        
    try:
        req = auth_pb2.ResetPasswordRequest(token=otp, email=email, new_password=new_password)
        response = stub.ResetPassword(req)
        if response.success:
            st.success(response.message)
            st.session_state.auth_page = "login"
            st.session_state.pop("reset_email", None)
            st.rerun()
        else:
            st.error(f"Failed to reset password: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.details()}")

# --- UI RENDERING FUNCTIONS ---
def render_login_form():
    st.title("🔐 Secure Login")
    oauth_component = OAuth2Component(
        client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        refresh_token_endpoint=None, revoke_token_endpoint="https://oauth2.googleapis.com/revoke",
    )
    auth_result = oauth_component.authorize_button(
        name="Continue with Google", icon="https://www.google.com/favicon.ico",
        redirect_uri=REDIRECT_URI, scope="openid email profile", key="google_login", use_container_width=True,
    )
    if auth_result and "token" in auth_result:
        handle_oauth_login(auth_result["token"])

    st.markdown("<p style='text-align: center; color: grey;'>or</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        st.text_input("Email", placeholder="your.email@example.com", key="login_email")
        st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Login with Email", use_container_width=True):
            handle_login()

    if st.button("Forgot Password?", use_container_width=True, type="secondary"):
        st.session_state.auth_page = "forgot_password"
        st.rerun()

    st.markdown("---")
    st.write("Don't have an account?")
    col1, col2 = st.columns(2)
    if col1.button("Create a User Account", use_container_width=True):
        st.session_state.auth_page = "register_user"
        st.rerun()
    if col2.button("Request Admin Access", use_container_width=True):
        st.session_state.auth_page = "register_admin"
        st.rerun()

def render_user_registration_form():
    st.subheader("📝 Create a User Account")
    with st.form("user_register_form"):
        st.text_input("Email", key="user_reg_email")
        st.text_input("Username", key="user_reg_username")
        # MODIFIED: Using the new phone number component
        st_phonenumbers_input("Phone Number (Optional)", key="user_reg_phone")
        st.text_input("Password", type="password", key="user_reg_password")
        # NEW: Added password requirements caption
        st.caption("Password must be 8+ characters and include an uppercase, lowercase, number, and special character.")
        if st.form_submit_button("Register", use_container_width=True):
            handle_user_register()
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()

def render_admin_registration_form():
    st.subheader("👑 Request Admin Access")
    with st.form("admin_register_form"):
        st.text_input("Full Name", key="admin_reg_fullname")
        st.text_input("Work Email", key="admin_reg_email")
        st.text_input("Username", key="admin_reg_username")
        st.text_input("Password", type="password", key="admin_reg_password")
        # NEW: Added password requirements caption
        st.caption("Password must be 8+ characters and include an uppercase, lowercase, number, and special character.")
        st.text_input("Employee ID (Optional)", key="admin_reg_empid")
        st.text_area("Reason for Request", key="admin_reg_reason")
        if st.form_submit_button("Submit Request", use_container_width=True):
            handle_admin_register()
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()

def render_otp_verification_form():
    st.subheader("✉️ Email Verification")
    user_email = st.session_state.get("pending_registration_email")
    if not user_email:
        st.error("Session error. Please try registering again.")
        st.session_state.auth_page = "register_user"
        if st.button("Go to Registration"):
            st.rerun()
        return

    st.info(f"An OTP has been sent to **{user_email}**. Please enter it below.")
    with st.form("otp_form"):
        st.text_input("Enter 6-digit OTP", max_chars=6, key="otp_code")
        if st.form_submit_button("Verify Account", use_container_width=True):
            handle_otp_verification()
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.session_state.pending_registration_email = None
        st.rerun()

def render_forgot_password_form():
    st.subheader("🔑 Forgot Password")
    st.info("Enter your email address to receive a password reset OTP.")
    with st.form("forgot_password_form"):
        st.text_input("Email", key="forgot_email")
        if st.form_submit_button("Send OTP", use_container_width=True):
            handle_forgot_password()
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()

def render_reset_password_form():
    st.subheader("🔒 Set New Password")
    email = st.session_state.get("reset_email")
    if not email:
        st.error("Session error. Please go back and request a new password reset OTP.")
        if st.button("Back to Forgot Password"):
            st.session_state.auth_page = "forgot_password"
            st.rerun()
        return

    st.info(f"Enter the OTP sent to **{email}** and create a new password.")
    with st.form("reset_password_form"):
        st.text_input("6-Digit OTP", key="reset_otp")
        st.text_input("New Password", type="password", key="reset_new_password")
        st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
        # NEW: Added password requirements caption
        st.caption("Password must be 8+ characters and include an uppercase, lowercase, number, and special character.")
        if st.form_submit_button("Reset Password", use_container_width=True):
            handle_reset_password()
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()

# --- PAGE ROUTING LOGIC ---
page = st.session_state.get("auth_page", "login")

if page == "login":
    render_login_form()
elif page == "register_user":
    render_user_registration_form()
elif page == "register_admin":
    render_admin_registration_form()
elif page == "verify_otp":
    render_otp_verification_form()
elif page == "forgot_password":
    render_forgot_password_form()
elif page == "reset_password":
    render_reset_password_form()
else:
    st.session_state.auth_page = "login"
    st.rerun()