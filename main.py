# main.py
import time
from datetime import datetime
import pandas as pd

from config import CSV_PATH
from classifier import train_and_report_classifier
from regressor import train_and_report_regressor
from visualize import plot_feature_importance
from timeseries import forecast_overdue_volume

def main():
    start = time.time()
    print(f"\n🚀 Job started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load your dataset
    df = pd.read_csv(CSV_PATH)

    # Run your models
    clf, X_test, y_test_class, y_pred_class = train_and_report_classifier(df)
    reg, X_test_reg, y_test_reg, y_pred_reg = train_and_report_regressor(df)

    # Visualize feature importances
    plot_feature_importance(clf, X_test.columns, title="Classifier: Feature Importance")
    plot_feature_importance(reg, X_test_reg.columns, title="Regressor: Feature Importance")

    # Run time series forecast
    forecast_overdue_volume(df)

    end = time.time()
    elapsed = end - start

    print(f"\n✅ Job completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱ Total elapsed time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
