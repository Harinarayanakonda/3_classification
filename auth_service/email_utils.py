import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# The dotenv import and load_dotenv() call have been removed.
# Environment variables are now loaded once by the main server.py script.

# --- Email Configuration ---
# These will correctly read the variables loaded by the main server script.
SMTP_SERVER = os.getenv("EMAIL_HOST")
SMTP_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_SENDER = os.getenv("EMAIL_HOST_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD")

# --- Generic Email Sending Function ---
def _send_email(recipient_email: str, subject: str, html_content: str):
    """
    A generic function to connect to the SMTP server and send an HTML email.
    """
    if not all([SMTP_SERVER, SMTP_PORT, EMAIL_SENDER, EMAIL_PASSWORD]):
        logging.error("Email configuration is incomplete. Check your credentials.env file.")
        return False, "Email service is not configured on the server."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Your App Name <{EMAIL_SENDER}>"
        msg["To"] = recipient_email
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logging.info(f"Email sent successfully to {recipient_email} with subject '{subject}'")
        return True, "Email sent successfully."

    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication Error. Check EMAIL_HOST_USER and EMAIL_HOST_PASSWORD in your .env file.")
        return False, "Failed to send email due to authentication error."
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending email: {e}")
        return False, f"An unexpected error occurred: {e}"

# --- Specific Email Functions ---

def send_otp_email(recipient_email: str, username: str, otp: str):
    """
    Sends a welcome email with a 6-digit OTP for account verification.
    """
    subject = "Your Verification Code"
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f4f7; padding: 20px; color: #333; }}
            .container {{ max-width: 600px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .header {{ font-size: 24px; font-weight: 600; color: #1d4ed8; text-align: center; margin-bottom: 20px; }}
            .otp-code {{ font-size: 36px; font-weight: bold; color: #1d4ed8; text-align: center; margin: 25px 0; letter-spacing: 5px; background-color: #eff6ff; padding: 15px; border-radius: 5px; }}
            p {{ line-height: 1.6; font-size: 16px; }}
            .footer {{ font-size: 12px; color: #777; margin-top: 20px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Account Verification</div>
            <p>Hello, <strong>{username}</strong>!</p>
            <p>Thank you for registering. Please use the following One-Time Password (OTP) to verify your account. The code is valid for 10 minutes.</p>
            <div class="otp-code">{otp}</div>
            <p>If you did not request this code, you can safely ignore this email.</p>
        </div>
        <div class="footer"><p>This is an automated message. Please do not reply.</p></div>
    </body>
    </html>
    """
    return _send_email(recipient_email, subject, html_content)


def send_password_reset_email(recipient_email: str, reset_link: str):
    """
    Sends an email with a link to reset the user's password.
    """
    subject = "Your Password Reset Request"
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f4f7; padding: 20px; color: #333; }}
            .container {{ max-width: 600px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center; }}
            .header {{ font-size: 24px; font-weight: 600; color: #1d4ed8; margin-bottom: 20px; }}
            p {{ line-height: 1.6; font-size: 16px; }}
            .button {{ background-color: #2563eb; color: white !important; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block; margin-top: 20px; }}
            .footer {{ font-size: 12px; color: #777; margin-top: 20px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Password Reset Request</div>
            <p>We received a request to reset your password. Click the button below to set a new one. This link will expire in 1 hour.</p>
            <a href="{reset_link}" class="button">Reset Password</a>
            <p style="margin-top: 30px;">If you did not request a password reset, you can safely ignore this email.</p>
        </div>
        <div class="footer"><p>This is an automated message. Please do not reply.</p></div>
    </body>
    </html>
    """
    return _send_email(recipient_email, subject, html_content)


def send_admin_approval_email(recipient_email: str):
    """
    Notifies a user that their request for admin access has been approved.
    """
    subject = "Admin Access Approved"
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f4f7; padding: 20px; color: #333; }}
            .container {{ max-width: 600px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .header {{ font-size: 24px; font-weight: 600; color: #16a34a; margin-bottom: 20px; }}
            p {{ line-height: 1.6; font-size: 16px; }}
            .footer {{ font-size: 12px; color: #777; margin-top: 20px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">🎉 Congratulations!</div>
            <p>Your request for administrator access on our platform has been approved.</p>
            <p>You can now log in to your account to access the admin dashboard and its features.</p>
            <p>Thank you for your contribution to our team.</p>
        </div>
        <div class="footer"><p>This is an automated message. Please do not reply.</p></div>
    </body>
    </html>
    """
    return _send_email(recipient_email, subject, html_content)