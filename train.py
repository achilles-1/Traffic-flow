import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

DATA_PATH = "preprocess.csv"
MODEL_PATH = "model.pkl"
df = pd.read_csv(DATA_PATH)
features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all"
]
X = df[features]
y = df["traffic_volume"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train,
    y_train
)
prediction = model.predict(X_test)
mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = mean_squared_error(
    y_test,
    prediction
) ** 0.5

r2 = r2_score(
    y_test,
    prediction
)


model_data = {
    "model": model,
    "features": features
}


joblib.dump(
    model_data,
    MODEL_PATH
)


print()
print("Model training completed")
print()
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))
print("R2 Performance:", round(r2 * 100, 2), "%")
print()
print("Model saved to:", MODEL_PATH)
