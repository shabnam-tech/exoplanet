import pandas as pd
from db import get_connection
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
features = joblib.load(os.path.join(BASE_DIR, "models", "model_features.pkl"))

conn = get_connection()
df = pd.read_sql_query('SELECT * FROM planets', conn)
conn.close()

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Features: {features}")
print("\nFirst row:")
if len(df) > 0:
    print(df.iloc[0])
print(f"\nData types:\n{df.dtypes}")
