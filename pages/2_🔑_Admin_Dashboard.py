# pages/2_🔑_Admin_Dashboard.py

import streamlit as st
import psutil
import psycopg2
from psycopg2 import pool
import grpc
import os
import redis
import json

from utils.ui_components import render_navbar
from auth_service import auth_pb2, auth_pb2_grpc
from utils.model_handler import load_model_artifacts

# --- SECURITY & NAVIGATION ---
# 1. Check if user is authenticated
if not st.session_state.get("authenticated", False):
    st.error("🚫 You must log in first!")
    st.stop()

# 2. Check if user is an ADMIN
role_enum = st.session_state.get("role")
role_name = auth_pb2.UserRole.Name(role_enum) if role_enum is not None else ""
if role_name != 'ADMIN':
    st.error("🚫 You do not have permission to view this page.")
    st.stop()

# 3. Render the navigation bar
render_navbar()
# --------------------------------

st.title("🔑 Admin Dashboard")
st.markdown("Monitor system health, manage models, and approve new admin accounts.")
st.markdown("---")

# --- DATABASE, GRPC & REDIS CONNECTIONS ---
@st.cache_resource
def init_db_connection():
    try:
        db_config = st.secrets["database"]
        return psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=10, **db_config
        )
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_resource
def init_redis_connection():
    try:
        redis_config = st.secrets["redis"]
        return redis.Redis(**redis_config, decode_responses=True)
    except Exception as e:
        st.error(f"Redis connection failed: {e}")
        return None

@st.cache_resource
def get_grpc_stub():
    try:
        channel = grpc.insecure_channel("localhost:50051")
        grpc.channel_ready_future(channel).result(timeout=5)
        return auth_pb2_grpc.AuthServiceStub(channel)
    except Exception as e:
        st.error(f"Failed to connect to gRPC server: {e}")
        return None

db_pool = init_db_connection()
grpc_stub = get_grpc_stub()
redis_conn = init_redis_connection()

if not db_pool or not grpc_stub or not redis_conn:
    st.stop()

# --- 1. OBSERVABILITY & MONITORING ---
st.header("📊 System Observability")

def get_user_stats(pool):
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
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    return cpu_usage, memory_info, disk_info

cpu, mem, disk = get_system_stats()
total_users, total_admins = get_user_stats(db_pool)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Active Model")
    if 'artifacts' in st.session_state and st.session_state.get('artifacts'):
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

# --- 2. MODEL MANAGEMENT ---
st.header("🚀 Model Management")

def get_model_files(directory="trained_models"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = [f for f in os.listdir(directory) if f.endswith(('.joblib', '.pkl'))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files

with st.expander("Activate a Model for Inference", expanded=False):
    model_files = get_model_files()
    if not model_files:
        st.warning("No trained models found in the 'trained_models' directory.")
    else:
        selected_model_file = st.selectbox(
            "Choose a model file to activate", model_files
        )
        if st.button("Load and Activate Model", use_container_width=True):
            with st.spinner(f"Loading {selected_model_file}..."):
                artifacts = load_model_artifacts(selected_model_file)
                if artifacts:
                    st.session_state['artifacts'] = artifacts
                    st.success(f"Model `{selected_model_file}` is now active for inference.")
                    st.rerun()

st.markdown("---")

# --- 3. USER MANAGEMENT (ADMIN APPROVAL) ---
st.header("👥 User Management")
st.subheader("Pending Admin Approvals")

def fetch_pending_admins(_stub, _redis):
    cache_key = "pending_admins_list"
    try:
        cached_admins = _redis.get(cache_key)
        if cached_admins:
            return json.loads(cached_admins)

        response = _stub.GetPendingAdmins(auth_pb2.EmptyRequest())
        # Updated to fetch all fields from the proto definition
        admins_list = [
            {
                "user_id": a.user_id, "username": a.username, "email": a.email,
                "full_name": a.full_name, "admin_request_reason": a.admin_request_reason,
                "created_at": a.created_at
            } for a in response.admins
        ]
        
        _redis.setex(cache_key, 600, json.dumps(admins_list))
        return admins_list
    except grpc.RpcError as e:
        st.error(f"gRPC Error: {e.details()}")
        return []
    except redis.exceptions.RedisError as e:
        st.error(f"Redis Error: {e}")
        response = _stub.GetPendingAdmins(auth_pb2.EmptyRequest())
        admins_list = [
            {
                "user_id": a.user_id, "username": a.username, "email": a.email,
                "full_name": a.full_name, "admin_request_reason": a.admin_request_reason,
                "created_at": a.created_at
            } for a in response.admins
        ]
        return admins_list

# MODIFIED: approve_admin now accepts expiry_days and handles cache invalidation
def approve_admin(stub, _redis, user_id, username, expiry_days):
    cache_key = "pending_admins_list"
    try:
        request = auth_pb2.AdminApprovalRequest(user_id=user_id, expiry_days=expiry_days)
        response = stub.ApproveAdmin(request)
        if response.success:
            st.toast(f"✅ Approved: {username}")
            _redis.delete(cache_key)
        else:
            st.error(f"Failed to approve {username}: {response.message}")
    except grpc.RpcError as e:
        st.error(f"gRPC Error: {e.details()}")

pending_admins = fetch_pending_admins(grpc_stub, redis_conn)

if not pending_admins:
    st.info("There are no pending admin requests.")
else:
    st.warning("The following users require approval to gain admin access.")
    
    # MODIFIED: Column layout updated for the new expiry input field
    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([2, 3, 2.5, 2, 1.5])
    h_col1.markdown("**Username**")
    h_col2.markdown("**Email**")
    h_col3.markdown("**Registered On**")
    h_col4.markdown("**Expiry (days)**")
    h_col5.markdown("**Action**")

    for user in pending_admins:
        # MODIFIED: Columns updated to match the new header layout
        cols = st.columns([2, 3, 2.5, 2, 1.5])
        cols[0].write(f"**{user['username']}**")
        cols[1].write(user['email'])
        cols[2].write(f"{user['created_at'].split('T')[0]}")
        
        # NEW: Added a number input for setting the expiry duration
        expiry_days = cols[3].number_input(
            "Days until expiry", 
            min_value=0, 
            value=90,  # Default to 90 days
            step=30, 
            key=f"expiry_{user['user_id']}", 
            label_visibility="collapsed"
        )
        
        if cols[4].button("Approve", key=f"approve_{user['user_id']}", use_container_width=True):
            # MODIFIED: Pass the expiry_days value to the approve function
            approve_admin(grpc_stub, redis_conn, user['user_id'], user['username'], expiry_days)
            st.rerun()