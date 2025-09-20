import streamlit as st
import os
import pandas as pd
import psutil
import psycopg2
from psycopg2 import pool
import grpc

# --- Local Imports ---
# Assumes 'utils' and 'auth_service' are in the project root
from utils.ui_components import render_navbar
from utils.model_handler import load_model_artifacts
from auth_service import auth_pb2, auth_pb2_grpc

# --- SECURITY & NAVIGATION ---
if not st.session_state.get("authenticated", False):
    st.error("🚫 You must log in first!")
    st.stop()

if st.session_state.get("role") != "admin":
    st.error("🚫 You do not have permission to view this page.")
    st.stop()

render_navbar()
# -----------------------------

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Admin Dashboard", layout="wide", page_icon="🔑")
st.title("🔑 Admin Dashboard")
st.markdown("Monitor system health, manage models, and approve new admin accounts.")
st.markdown("---")

# --- DATABASE & GRPC CONNECTIONS ---

@st.cache_resource
def init_db_connection():
    """Initializes a connection pool to the PostgreSQL database."""
    try:
        db_config = st.secrets["database"]
        return psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=10,
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config.get("port", 5432)
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_resource
def get_grpc_stub():
    """Creates and caches the gRPC stub for server communication."""
    try:
        channel = grpc.insecure_channel("localhost:50051")
        grpc.channel_ready_future(channel).result(timeout=5)
        return auth_pb2_grpc.AuthServiceStub(channel)
    except Exception as e:
        st.error(f"Failed to connect to gRPC server: {e}")
        return None

db_pool = init_db_connection()
grpc_stub = get_grpc_stub()

if not db_pool or not grpc_stub:
    st.stop()

# ### --- 1. OBSERVABILITY & MONITORING --- ###
st.header("📊 System Observability")

def get_user_stats(pool):
    """Fetches user and admin counts from the database."""
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users;")
            total_users = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND status = 'active';")
            total_admins = cur.fetchone()[0]
            return total_users, total_admins
    except Exception as e:
        st.error(f"Failed to fetch user stats: {e}")
        return 0, 0
    finally:
        if conn:
            pool.putconn(conn)

def get_system_stats():
    """Retrieves system resource usage."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    return cpu_usage, memory_info, disk_info

cpu, mem, disk = get_system_stats()
total_users, total_admins = get_user_stats(db_pool)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Active Model")
    if 'artifacts' in st.session_state and st.session_state['artifacts'] is not None:
        model_name = st.session_state['artifacts']['pipeline'].steps[-1][1].__class__.__name__
        class_labels = st.session_state['artifacts']['class_labels']
        st.success(f"**Model:** `{model_name}`", icon="✅")
        st.info(f"**Classes:** `{', '.join(class_labels)}`", icon="🏷️")
    else:
        st.warning("**No model is currently active.**", icon="⚠️")

with col2:
    st.subheader("Resource Usage")
    st.metric(label="CPU Usage", value=f"{cpu}%")
    st.metric(label="Memory Usage", value=f"{mem.percent}%")
    st.metric(label="Disk Usage", value=f"{disk.percent}%")

with col3:
    st.subheader("User Statistics")
    st.metric(label="Total Registered Users", value=total_users)
    st.metric(label="Total Active Admins", value=total_admins)

st.markdown("---")

# ### --- 2. MODEL MANAGEMENT (This section remains the same) --- ###
st.header("🚀 Model Management")

def get_model_files(directory="trained_models"):
    if not os.path.exists(directory):
        st.error(f"Directory not found: '{directory}'.")
        return []
    files = [f for f in os.listdir(directory) if f.endswith(('.joblib', '.pkl'))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files

with st.expander("Activate a Model for Inference", expanded=True):
    model_files = get_model_files()
    if not model_files:
        st.warning("No trained models found in the 'trained_models' directory.")
    else:
        selected_model_file = st.selectbox(
            "Choose a model file to activate", model_files, index=0
        )
        if st.button("Load and Activate Model", use_container_width=True):
            with st.spinner(f"Loading {selected_model_file}..."):
                artifacts = load_model_artifacts(selected_model_file)
                if artifacts:
                    st.session_state['artifacts'] = artifacts
                    st.success(f"Model `{selected_model_file}` is now active for inference.")
                    st.rerun()

# ### --- 3. USER MANAGEMENT (ADMIN APPROVAL) --- ###
st.markdown("---")
st.header("👥 User Management")
st.subheader("Pending Admin Approvals")

def fetch_pending_admins(stub):
    """Fetches pending admin users via gRPC."""
    try:
        request = auth_pb2.EmptyRequest()
        response = stub.GetPendingAdmins(request)
        return response.admins
    except grpc.RpcError as e:
        st.error(f"Failed to fetch pending admins: {e.details()}")
        return []

def approve_admin(stub, user_id, username):
    """Approves an admin via gRPC and shows a success message."""
    try:
        request = auth_pb2.AdminApprovalRequest(user_id=user_id)
        response = stub.ApproveAdmin(request)
        if response.success:
            st.toast(f"✅ Approved admin access for {username}.")
        else:
            st.error(f"Failed to approve {username}: {response.message}")
    except grpc.RpcError as e:
        st.error(f"RPC error approving admin: {e.details()}")

pending_admins = fetch_pending_admins(grpc_stub)

if not pending_admins:
    st.info("There are no pending admin requests at this time.")
else:
    st.warning("The following users have registered as admins and require approval.")
    
    for user in pending_admins:
        col1, col2, col3, col4 = st.columns([2, 3, 3, 1.5])
        with col1:
            st.write(f"**{user.username}**")
        with col2:
            st.write(user.email)
        with col3:
            # The 'created_at' from proto is a string, display it
            st.write(f"Registered: {user.created_at.split('T')[0]}")
        with col4:
            st.button(
                "Approve", 
                key=f"approve_{user.user_id}", 
                on_click=approve_admin,
                args=(grpc_stub, user.user_id, user.username), # Pass necessary arguments
                use_container_width=True
            )