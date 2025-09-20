import streamlit as st
import grpc
import sys
import os

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(page_title="Authentication | ML App", layout="centered")
st.markdown("<style>[data-testid='stSidebar'] {display: none;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)


from auth_service import auth_pb2, auth_pb2_grpc

@st.cache_resource
def get_grpc_stub():
    """Creates and caches the gRPC stub for server communication."""
    try:
        channel = grpc.insecure_channel("localhost:50051")
        grpc.channel_ready_future(channel).result(timeout=5)
        return auth_pb2_grpc.AuthServiceStub(channel)
    except grpc.FutureTimeoutError:
        st.error("Connection to the auth server timed out. Please ensure the gRPC server is running.")
        return None
    except Exception as e:
        st.error(f"A gRPC connection error occurred: {e}")
        return None

stub = get_grpc_stub()
if not stub:
    st.stop()

# --- SESSION STATE & QUERY PARAMS ---
if 'authenticated' not in st.session_state:
    st.session_state.update({
        "authenticated": False, "token": None, "user_id": None, "role": None,
        "auth_page": "login", "pending_registration_email": None
    })

# Check for password reset query parameters on first load
query_params = st.query_params
if "page" in query_params and query_params["page"] == "reset_password":
    if "token" in query_params and "email" in query_params:
        st.session_state.auth_page = "reset_password"
        st.session_state.reset_token = query_params["token"]
        st.session_state.reset_email = query_params["email"]
        # Clear params to avoid loops
        st.query_params.clear()

# --- REDIRECTION LOGIC ---
if st.session_state.get("authenticated"):
    if st.session_state.role == 'admin':
        st.switch_page("pages/1_💾_Dataset_Upload_&Preprocessing.py")
    else:
        st.switch_page("pages/3_🔮_Inference.py")

# --- UI RENDERING FUNCTIONS ---
def render_login_form():
    st.title("🔐 Secure Login")
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password")
        st.form_submit_button("Login", on_click=handle_login, args=(email, password), use_container_width=True)

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

def render_forgot_password_form():
    st.subheader("🔑 Forgot Password")
    st.info("Enter your email address and we'll send you a link to reset your password.")
    with st.form("forgot_password_form"):
        email = st.text_input("Email")
        st.form_submit_button("Send Reset Link", on_click=handle_forgot_password, args=(email,), use_container_width=True)
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()

def render_reset_password_form():
    st.subheader("🔒 Set New Password")
    email = st.session_state.get("reset_email", "your email")
    st.info(f"Create a new password for **{email}**.")
    with st.form("reset_password_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Reset Password", use_container_width=True)
        if submitted:
            handle_reset_password(new_password, confirm_password)

# ... (Other render functions are the same) ...
def render_user_registration_form():
    st.subheader("📝 Create a User Account")
    with st.form("user_register_form"):
        email = st.text_input("Email")
        username = st.text_input("Username")
        phone_number = st.text_input("Phone Number (Optional)")
        password = st.text_input("Password", type="password")
        st.form_submit_button("Register", on_click=handle_user_register, args=(email, username, phone_number, password), use_container_width=True)
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.rerun()

def render_otp_verification_form():
    st.subheader("✉️ Email Verification")
    user_email = st.session_state.get("pending_registration_email")
    if not user_email:
        st.error("Session error. Please try registering again.")
        st.session_state.auth_page = "register_user"
        st.rerun()
        return

    st.info(f"An OTP has been sent to **{user_email}**. Please enter it below.")
    with st.form("otp_form"):
        otp = st.text_input("Enter 6-digit OTP", max_chars=6, key="otp_input")
        st.form_submit_button("Verify Account", on_click=handle_otp_verification, args=(user_email, otp), use_container_width=True)
    if st.button("Back to Login"):
        st.session_state.auth_page = "login"
        st.session_state.pending_registration_email = None
        st.rerun()

# --- HANDLER FUNCTIONS ---
# ... (handle_login, handle_user_register, handle_otp_verification are the same) ...
def handle_login(email, password):
    if not all([email, password]):
        st.warning("Please enter both email and password.")
        return
    try:
        response = stub.Login(auth_pb2.LoginRequest(email=email, password=password))
        if response.success:
            st.session_state.authenticated = True
            st.session_state.token = response.token
            st.session_state.user_id = response.user_id
            st.session_state.role = response.role
            st.rerun()
        else:
            st.error(f"Login Failed: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.status().details()}")

def handle_user_register(email, username, phone, password):
    if not all([email, username, password]):
        st.warning("Please fill all required fields.")
        return
    try:
        req = auth_pb2.RegisterUserRequest(email=email, username=username, password=password, phone_number=phone)
        response = stub.RegisterUser(req)
        if response.success:
            st.session_state.pending_registration_email = email
            st.session_state.auth_page = "verify_otp"
            st.rerun()
        else:
            st.error(f"Registration Failed: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.status().details()}")

def handle_otp_verification(email, otp):
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
        st.error(f"Server communication error: {e.status().details()}")

def handle_forgot_password(email):
    if not email:
        st.warning("Please enter your email address.")
        return
    try:
        response = stub.ForgotPassword(auth_pb2.ForgotPasswordRequest(email=email))
        if response.success:
            st.success(response.message)
            st.session_state.auth_page = "login"
        else:
            st.error(f"Error: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.status().details()}")

def handle_reset_password(new_password, confirm_password):
    if not all([new_password, confirm_password]):
        st.error("Please fill out both password fields.")
        return
    if new_password != confirm_password:
        st.error("Passwords do not match.")
        return
    
    token = st.session_state.get("reset_token")
    email = st.session_state.get("reset_email")

    try:
        req = auth_pb2.ResetPasswordRequest(token=token, email=email, new_password=new_password)
        response = stub.ResetPassword(req)
        if response.success:
            st.success(response.message)
            st.session_state.auth_page = "login"
            st.session_state.reset_token = None
            st.session_state.reset_email = None
            st.rerun()
        else:
            st.error(f"Failed to reset password: {response.message}")
    except grpc.RpcError as e:
        st.error(f"Server communication error: {e.status().details()}")

# --- PAGE ROUTING LOGIC ---
page = st.session_state.get("auth_page", "login")

if page == "login":
    render_login_form()
elif page == "register_user":
    render_user_registration_form()
elif page == "verify_otp":
    render_otp_verification_form()
elif page == "forgot_password":
    render_forgot_password_form()
elif page == "reset_password":
    render_reset_password_form()
else:
    render_login_form()