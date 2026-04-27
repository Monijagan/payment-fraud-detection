# Payment Fraud Detection System

End-to-end fraud detection pipeline for P2P payment transactions.

## Tech Stack
- Python (Pandas, XGBoost, Scikit-Learn)
- PostgreSQL (Data Warehouse)
- Power BI (Dashboard)
- SQL (Feature Engineering)

## Project Structure
- `Scripts/etl_pipeline.py` - ETL pipeline loading data into PostgreSQL
- `Scripts/ml_model.py` - XGBoost fraud detection model
- `Dataset/` - Synthetic transaction data

## Results
- 50,000 transactions analyzed
- 73.5% fraud detection recall
- 23 engineered features
- Real-time risk scoring (0-1000 scale)
