# AR Simulator

Simulated ML modeling of synthetic defense contracting data. Uses scikit-learn to explore patterns in overdue accounts receivable. Great for testing classification, regression, and time series on randomized financial scenarios.

Part of the Regulus fintech suite.

## Usage

🔧 How to Use ar-forecast

1. Clone the repository:

git clone https://github.com/Hamiltonius/AR-Forecast.git
cd AR-Forecast

2. (Optional, but recommended) Create a virtual environment and install dependencies:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

3. Load the included dataset:
The repo comes with a synthetic dataset:
📄 defense_ar_synthetic_data.csv

If you’d like to use your own data, just replace this file and ensure the format matches what’s expected in dataload.py.

4. Run the main script:

python main.py

This will:
	•	Load the dataset
	•	Train classification and regression models on overdue behavior
	•	Output feature importance and forecast charts to interpret trends and AR risk