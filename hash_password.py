import bcrypt

# --- ENTER THE PASSWORD YOU WANT FOR YOUR SUPER ADMIN HERE ---
# The 'b' before the string is important.
password_to_hash = b"6"

# Generate the secure hash
hashed_password = bcrypt.hashpw(password_to_hash, bcrypt.gensalt())

# Print the result
print("✅ Your hashed password is ready.")
print("Copy this entire line and paste it into your SQL command:")
print(hashed_password.decode())