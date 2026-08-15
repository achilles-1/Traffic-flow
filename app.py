from flask import Flask, render_template, request
import joblib
import pandas as pd

from optimize_traffic import optimize_traffic


app = Flask(__name__)


# ==============================
# Load Regression Model
# ==============================

MODEL_PATH = "model.pkl"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]


# ==============================
# Load Classification Model
# ==============================

CLASSIFIER_PATH = "traffic_classifier.pkl"

classifier_data = joblib.load(CLASSIFIER_PATH)

classifier = classifier_data["model"]
classifier_features = classifier_data["features"]


# ==============================
# Model Performance
# ==============================

MODEL_ACCURACY = 95.13
R2_SCORE = 0.9513
MAE = 234.40
RMSE = 439.01


# ==============================
# Home Page
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    traffic_level = None
    green_time = None
    red_time = None

    if request.method == "POST":

        # ==============================
        # Get Input Values
        # ==============================

        hour = int(
            request.form["hour"]
        )

        day_of_week = int(
            request.form["day_of_week"]
        )

        month = int(
            request.form["month"]
        )

        temperature = float(
            request.form["temperature"]
        )

        rain = float(
            request.form["rain"]
        )

        snow = float(
            request.form["snow"]
        )

        clouds = float(
            request.form["clouds"]
        )

        holiday = int(
            request.form["holiday"]
        )


        # ==============================
        # Calculate Derived Features
        # ==============================

        is_weekend = int(
            day_of_week >= 5
        )

        is_peak_hour = int(
            hour in [7, 8, 9, 16, 17, 18]
        )


        # ==============================
        # Create Input Data
        # ==============================

        input_data = pd.DataFrame(
            [[
                hour,
                day_of_week,
                month,
                is_weekend,
                is_peak_hour,
                holiday,
                temperature,
                rain,
                snow,
                clouds
            ]],
            columns=features
        )


        # ==============================
        # Predict Traffic Volume
        # ==============================

        prediction = model.predict(
            input_data
        )[0]


        # ==============================
        # Classify Traffic Level
        # ==============================

        traffic_level = classifier.predict(
            input_data
        )[0]


        # ==============================
        # Optimize Traffic
        # ==============================

        green_time, red_time = optimize_traffic(
            traffic_level
        )


    # ==============================
    # Send Data to HTML
    # ==============================

    return render_template(
        "index.html",
        prediction=prediction,
        traffic_level=traffic_level,
        green_time=green_time,
        red_time=red_time,
        model_accuracy=MODEL_ACCURACY,
        r2_score=R2_SCORE,
        mae=MAE,
        rmse=RMSE
    )


# ==============================
# Run Flask Application
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )