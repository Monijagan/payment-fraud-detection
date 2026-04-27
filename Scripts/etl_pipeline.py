"""
ETL Pipeline: Load Payment Fraud Data into PostgreSQL
Extracts CSV files → Transforms/Cleans → Loads into database
"""

import pandas as pd
import psycopg2
from psycopg2 import sql
import os

print("Starting ETL Pipeline...")
print("=" * 60)

# Database connection settings
DB_CONFIG = {
    'host': 'localhost',
    'database': 'postgres',
    'user': 'postgres',
    'password': '250499',  # Change this to your PostgreSQL password
    'port': 5433
}

# File paths - CHANGE THESE to where you saved the CSV files
DATA_DIR = "C:/Users/monij/OneDrive/Documents/Fraud_analytics/Dataset/"  # Windows

print("Step 1/4: Connecting to PostgreSQL...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✓ Connected to database")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\nMake sure PostgreSQL is running!")
    exit(1)

# Step 2: Create tables
print("\nStep 2/4: Creating database tables...")

# Drop tables if they exist (fresh start)
cursor.execute("DROP TABLE IF EXISTS transactions CASCADE;")
cursor.execute("DROP TABLE IF EXISTS customers CASCADE;")
cursor.execute("DROP TABLE IF EXISTS recipients CASCADE;")

# Create customers table
cursor.execute("""
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    account_age_days INTEGER,
    city VARCHAR(50),
    state VARCHAR(2),
    risk_tier VARCHAR(10)
);
""")

# Create recipients table
cursor.execute("""
CREATE TABLE recipients (
    recipient_id VARCHAR(20) PRIMARY KEY,
    recipient_name VARCHAR(100),
    recipient_email VARCHAR(100),
    recipient_phone VARCHAR(20),
    recipient_account_age_days INTEGER,
    recipient_verified BOOLEAN
);
""")

# Create transactions table
cursor.execute("""
CREATE TABLE transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES customers(customer_id),
    recipient_id VARCHAR(20) REFERENCES recipients(recipient_id),
    timestamp TIMESTAMP,
    amount NUMERIC(10,2),
    description VARCHAR(100),
    device_id VARCHAR(100),
    is_first_time_recipient BOOLEAN,
    status VARCHAR(20),
    is_fraud BOOLEAN,
    fraud_type VARCHAR(30),
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_night BOOLEAN,
    is_round_dollar BOOLEAN
);
""")

conn.commit()
print("✓ Tables created")

# Step 3: Load data from CSV files
print("\nStep 3/4: Loading CSV files...")

# Load customers
customers_df = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'))
print(f"  Loading {len(customers_df)} customers...")
for _, row in customers_df.iterrows():
    cursor.execute("""
        INSERT INTO customers VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

# Load recipients
recipients_df = pd.read_csv(os.path.join(DATA_DIR, 'recipients.csv'))
print(f"  Loading {len(recipients_df)} recipients...")
for _, row in recipients_df.iterrows():
    cursor.execute("""
        INSERT INTO recipients VALUES (%s, %s, %s, %s, %s, %s)
    """, tuple(row))

# Load transactions
transactions_df = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'))
print(f"  Loading {len(transactions_df)} transactions...")
for _, row in transactions_df.iterrows():
    cursor.execute("""
        INSERT INTO transactions VALUES 
        (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()
print("✓ Data loaded")

# Step 4: Verify data
print("\nStep 4/4: Verifying data load...")
cursor.execute("SELECT COUNT(*) FROM customers;")
print(f"  Customers: {cursor.fetchone()[0]:,}")

cursor.execute("SELECT COUNT(*) FROM recipients;")
print(f"  Recipients: {cursor.fetchone()[0]:,}")

cursor.execute("SELECT COUNT(*) FROM transactions;")
print(f"  Transactions: {cursor.fetchone()[0]:,}")

cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = true;")
print(f"  Fraud cases: {cursor.fetchone()[0]:,}")

print("\n" + "=" * 60)
print("ETL PIPELINE COMPLETE! ✅")
print("=" * 60)
print("\nDatabase ready for Hour 3: Feature Engineering & ML Model")

cursor.close()
conn.close()