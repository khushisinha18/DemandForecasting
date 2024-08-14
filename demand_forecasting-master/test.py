# test_prophet.py
try:
    from prophet import Prophet
    print("Prophet imported successfully!")
except ModuleNotFoundError as e:
    print("Error importing Prophet:", e)


