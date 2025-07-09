import pandas as pd

from config import RANDOM_SEED, TEST_SIZE, N_ESTIMATORS
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_report_classifier(df):
    df['month'] = pd.to_datetime(df['entry_date']).dt.month
    df['is_overdue'] = df['status'].apply(lambda x: 1 if x == 'OVERDUE' else 0)

    features = df[['total_value', 'month']]
    target = df['is_overdue']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    clf = XGBClassifier(n_estimators=N_ESTIMATORS, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    return clf, X_test, y_test, y_pred
