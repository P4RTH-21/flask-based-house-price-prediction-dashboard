from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and pipeline once when Flask starts
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from form

        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        housing_median_age = float(request.form["housing_median_age"])
        total_rooms = float(request.form["total_rooms"])
        total_bedrooms = float(request.form["total_bedrooms"])
        population = float(request.form["population"])
        households = float(request.form["households"])
        median_income = float(request.form["median_income"])
        ocean_proximity = request.form["ocean_proximity"]

        # Create DataFrame

        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
        }])

        # Apply preprocessing pipeline
        transformed_data = pipeline.transform(input_data)

        # Make prediction
        prediction = model.predict(transformed_data)

        return render_template(
            "index.html",
            prediction=f"${prediction[0]:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)