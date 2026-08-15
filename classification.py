import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


DATA_PATH = "preprocess.csv"
MODEL_PATH = "traffic_classifier.pkl"


# =========================
# Load processed dataset
# =========================

df = pd.read_csv(DATA_PATH)


# =========================
# Create traffic categories
# =========================

def classify_traffic(volume):

    if volume < 3000:
        return "Low"

    elif volume < 6000:
        return "Medium"

    else:
        return "High"


df["traffic_level"] = df["traffic_volume"].apply(
    classify_traffic
)


# =========================
# Features
# =========================

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

y = df["traffic_level"]


# =========================
# Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# =========================
# Random Forest Classifier
# =========================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# =========================
# Prediction
# =========================

prediction = model.predict(
    X_test
)


# =========================
# Evaluation
# =========================

accuracy = accuracy_score(
    y_test,
    prediction
)


print()
print("==============================")
print("TRAFFIC CLASSIFICATION MODEL")
print("==============================")

print()

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()

print("Classification Report:")

print(
    classification_report(
        y_test,
        prediction
    )
)


# =========================
# Save model
# =========================

model_data = {

    "model": model,

    "features": features

}


joblib.dump(
    model_data,
    MODEL_PATH
)


print()

print(
    "Model saved to:",
    MODEL_PATH
)