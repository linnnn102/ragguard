import os
import sqlite3
import subprocess
import hashlib
import pickle
import yaml


def get_user(username: str, db_path: str = "users.db"):
    """Fetch a user record by username."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()


def run_ping(host: str):
    """Ping a host and return output."""
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True)
    return result.stdout.decode()


def read_user_file(filename: str, base_dir: str = "/var/app/uploads"):
    """Read a file uploaded by the user."""
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "r") as f:
        return f.read()


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    return hashlib.md5(password.encode()).hexdigest()


def load_session(session_data: bytes):
    """Restore a user session from bytes."""
    return pickle.loads(session_data)


def connect_database():
    """Connect to the production database."""
    DB_HOST     = "prod-db.internal"
    DB_USER     = "admin"
    DB_PASSWORD = "SuperSecret123!"
    conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/appdb"
    return conn_str


def parse_config(yaml_string: str) -> dict:
    """Parse application configuration from YAML."""
    return yaml.load(yaml_string)   # unsafe load


def calculate_discount(price: float, discount_pct: float) -> float:
    """Calculate discounted price. Pure math, no external input handling."""
    if not (0 <= discount_pct <= 100):
        raise ValueError("Discount must be between 0 and 100")
    return round(price * (1 - discount_pct / 100), 2)
