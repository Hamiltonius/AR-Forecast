# timeseries.py
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def forecast_overdue_volume(df, periods=90):
    """
    Forecast the total overdue amount per day using Prophet.
    """
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    
    # Group by entry_date to simulate "daily overdue volume"
    daily_overdue = (
        df[df['status'] == 'OVERDUE']
        .groupby('entry_date')['overdue_amount']
        .sum()
        .reset_index()
        .rename(columns={'entry_date': 'ds', 'overdue_amount': 'y'})
    )

    # Initialize and fit model
    model = Prophet()
    model.fit(daily_overdue)

    # Forecast into the future
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Plot results
    fig1 = model.plot(forecast)
    plt.title("Forecast: Daily Overdue Amount")
    plt.xlabel("Date")
    plt.ylabel("USD ($)")
    plt.tight_layout()
    plt.show()

    return forecast
