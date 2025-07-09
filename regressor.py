# regressor.py
from config import RANDOM_SEED, TEST_SIZE, N_ESTIMATORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def train_and_report_regressor(df):
    df['month'] = pd.to_datetime(df['entry_date']).dt.month
    df['days_to_ship'] = (pd.to_datetime(df['po_ship_date']) - pd.to_datetime(df['entry_date'])).dt.days
    df['days_to_receive'] = (pd.to_datetime(df['po_receipt_date']) - pd.to_datetime(df['po_ship_date'])).dt.days

    df_overdue = df[df['status'] == 'OVERDUE'].copy()

    features = df_overdue[['total_value', 'days_to_ship', 'days_to_receive', 'month']]
    target = df_overdue['days_overdue']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    reg = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    print("\n=== REGRESSION REPORT (Days Overdue) ===")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"MSE:  {mean_squared_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R^2:  {r2_score(y_test, y_pred):.2f}")

    return reg, X_test, y_test, y_pred
