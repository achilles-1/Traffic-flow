from classification import classify_traffic
import joblib
import pandas as pd


MODEL_PATH = "model.pkl"


# Load trained model
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


print("===== Traffic Volume Prediction =====")


# Get input from user
hour = int(input("Enter hour (0-23): "))
day_of_week = int(input("Enter day of week (0=Monday, 6=Sunday): "))
month = int(input("Enter month (1-12): "))

temp = float(input("Enter temperature: "))
rain_1h = float(input("Enter rain in last 1 hour: "))
snow_1h = float(input("Enter snow in last 1 hour: "))
clouds_all = float(input("Enter cloudiness (0-100): "))

holiday_input = input("Is it a holiday? (yes/no): ").strip().lower()


# Convert holiday to 0 or 1
if holiday_input == "yes":
    holiday = 1
else:
    holiday = 0


# Calculate weekend automatically
is_weekend = 1 if day_of_week >= 5 else 0


# Calculate peak hour automatically
is_peak_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0


# Create input data
input_data = pd.DataFrame([{
    "hour": hour,
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": is_weekend,
    "is_peak_hour": is_peak_hour,
    "holiday": holiday,
    "temp": temp,
    "rain_1h": rain_1h,
    "snow_1h": snow_1h,
    "clouds_all": clouds_all
}])



# Make prediction
prediction = model.predict(input_data[features])[0]

# Classify predicted traffic
traffic_level = classify_traffic(prediction)

print()
print("===== Prediction Result =====")
print("Predicted Traffic Volume:", round(prediction, 2), "vehicles")
print("Traffic Level:", traffic_level)