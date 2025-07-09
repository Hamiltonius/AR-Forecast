# data loader
from config import CSV_PATH, RANDOM_SEED

import pandas as pd

df = pd.read_csv('/Users/tgalarneau2024/Github stuff/0415 Regulus/Branch_AR_Data/defense_ar_synthetic_data.csv')

# Convert date columns
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['po_ship_date'] = pd.to_datetime(df['po_ship_date'])
df['po_receipt_date'] = pd.to_datetime(df['po_receipt_date'])

# Add derived time-based features
df['days_to_ship'] = (df['po_ship_date'] - df['entry_date']).dt.days
df['days_to_receive'] = (df['po_receipt_date'] - df['po_ship_date']).dt.days
df['days_total'] = (df['po_receipt_date'] - df['entry_date']).dt.days
df['month'] = df['entry_date'].dt.month


df['overdue_flag'] = df['status'].apply(lambda x: 1 if x == 'OVERDUE' else 0)

features = df[['total_value', 'days_to_ship', 'days_to_receive', 'month']]
target_class = df['overdue_flag']
target_days = df['days_overdue']
