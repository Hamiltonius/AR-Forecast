import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed
np.random.seed(42)
random.seed(42)

# Config
num_entries = 500
start_date = datetime(2025, 1, 1)

# Customer list with story traits
customers = [
    {"name": "Lockheed", "late_behavior": "always_15_days_late"},
    {"name": "Raytheon", "late_behavior": "rarely_late"},
    {"name": "DCMA", "late_behavior": "quarterly_delays"},
    {"name": "Northrop", "late_behavior": "high_value_approval"},
    {"name": "Boeing", "late_behavior": "random"},
]

# Simulate entries
data = []
for _ in range(num_entries):
    cust = random.choice(customers)
    entry_date = start_date + timedelta(days=np.random.randint(0, 180))
    total_value = np.random.uniform(50000, 1000000)

    # Simulate delay logic
    if cust["late_behavior"] == "always_15_days_late":
        days_overdue = 15
    elif cust["late_behavior"] == "rarely_late":
        days_overdue = np.random.choice([0]*8 + [5, 10])
    elif cust["late_behavior"] == "quarterly_delays":
        if entry_date.month in [3, 6, 9, 12]:
            days_overdue = np.random.choice([0, 5, 10, 30])
        else:
            days_overdue = 0
    elif cust["late_behavior"] == "high_value_approval":
        if total_value > 800000:
            days_overdue = np.random.choice([30, 45, 60])
        else:
            days_overdue = 0
    else:  # random
        days_overdue = np.random.choice([0, 5, 10, 15, 20, 25, 30])

    status = "OVERDUE" if days_overdue > 0 else "PAID"
    po_ship_date = entry_date + timedelta(days=np.random.randint(2, 10))
    po_receipt_date = po_ship_date + timedelta(days=np.random.randint(1, 7))

    data.append({
        "customer_name": cust["name"],
        "total_value": round(total_value, 2),
        "entry_date": entry_date.strftime("%Y-%m-%d"),
        "po_ship_date": po_ship_date.strftime("%Y-%m-%d"),
        "po_receipt_date": po_receipt_date.strftime("%Y-%m-%d"),
        "days_overdue": days_overdue,
        "status": status
    })

# Save to DataFrame and export
df_story = pd.DataFrame(data)
output_path = "/mnt/data/defense_ar_story_data_fixed.csv"
df_story.to_csv(output_path, index=False)

import ace_tools as tools; tools.display_dataframe_to_user(name="Fixed Story-Based AR Dataset", dataframe=df_story)
output_path
