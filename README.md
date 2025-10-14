# Classification Platform

A Python-based platform for dataset upload, preprocessing, model training, and user authentication with admin management.

## Features

- **Dataset Upload & Preprocessing:** Upload datasets, preview, select target/features, and preprocess data ([pages/1_💾_Dataset_Upload_and_Preprocessing.py](pages/1_%F0%9F%92%BE_Dataset_Upload_and_Preprocessing.py)).
- **Model Training & Management:** Train models, tune hyperparameters, save/load models, and activate models for inference ([utils/model_handler.py](utils/model_handler.py), [pages/2_🔑_Admin_Dashboard.py](pages/2_%F0%9F%94%91_Admin_Dashboard.py)).
- **Authentication Service:** User/admin registration, login, OTP verification, password reset, and OAuth support via gRPC ([auth_service/server.py](auth_service/server.py), [auth_service/auth_pb2_grpc.py](auth_service/auth_pb2_grpc.py)).
- **Admin Dashboard:** Monitor system health, manage models, and approve admin accounts ([pages/2_🔑_Admin_Dashboard.py](pages/2_%F0%9F%94%91_Admin_Dashboard.py)).
- **Virtual Environment:** Scripts for activating Python virtual environments ([bin/Activate.ps1](bin/Activate.ps1)).

## Directory Structure

- `pages/` — Streamlit UI pages for dataset, model, and admin management.
- `auth_service/` — gRPC authentication microservice.
- `utils/` — Model handling and utility functions.
- `trained_models/` — Saved model files.
- `bin/` — Virtual environment scripts.
- `.streamlit/` — Streamlit configuration and secrets.
- `requirements.txt` — Python dependencies.

## Getting Started

1. **Clone the repository**
2. **Create and activate a virtual environment**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   Running All Services
Start Redis Server
Install Redis (if not already installed):

Start Redis:

Start Authentication Service (gRPC)

Start Streamlit App

Configure secrets.toml
Create or edit secrets.toml:
[redis]
host = "localhost"
port = 6379
db = 0

[auth]
grpc_host = "localhost"
grpc_port = 50051

[admin]
email = "your_admin_email@example.com"
password = "your_admin_password"

1 vulnerability
Configure credentials.env
Create or edit credentials.env:
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

GRPC_HOST=localhost
GRPC_PORT=50051

ADMIN_EMAIL=your_admin_email@example.com
ADMIN_PASSWORD=your_admin_password

Update your README with these instructions for clarity.