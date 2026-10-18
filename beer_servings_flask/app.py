
from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Load the saved model and preprocessing objects
model = joblib.load(BASE_DIR / "best_model.pkl")
encoder = joblib.load(BASE_DIR / "encoder.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
num_imputer = joblib.load(BASE_DIR / "num_imputer.pkl")
cat_imputer = joblib.load(BASE_DIR / "cat_imputer.pkl")
metadata = joblib.load(BASE_DIR / "model_metadata.pkl")

categorical_features = metadata["categorical_features"]
numerical_features = metadata["numerical_features"]
encoded_columns = metadata["encoded_columns"]

# Load data for dropdowns and landing-page infographics
data = pd.read_csv(BASE_DIR / "beer-servings.csv")
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

countries = metadata["countries"]
continents = metadata["continents"]

# Generate a chart for the landing page
def create_charts():
    chart_data = data.dropna(
        subset=["continent", "total_litres_of_pure_alcohol"]
    )

    averages = (
        chart_data.groupby("continent")[
            "total_litres_of_pure_alcohol"
        ]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(7, 4))
    averages.plot(kind="bar")
    plt.title("Average Pure Alcohol by Continent")
    plt.xlabel("Continent")
    plt.ylabel("Average litres")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "continent_chart.png", dpi=120)
    plt.close()


create_charts()


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    form_values = {
        "country": countries[0] if countries else "",
        "beer_servings": "",
        "spirit_servings": "",
        "wine_servings": "",
        "continent": continents[0] if continents else ""
    }

    if request.method == "POST":
        form_values = {
            "country": request.form.get("country", ""),
            "beer_servings": request.form.get("beer_servings", ""),
            "spirit_servings": request.form.get("spirit_servings", ""),
            "wine_servings": request.form.get("wine_servings", ""),
            "continent": request.form.get("continent", "")
        }

        try:
            country = form_values["country"]
            continent = form_values["continent"]

            # Validate dropdown selections
            if country not in countries:
                raise ValueError("Please select a valid country.")

            if continent not in continents:
                raise ValueError("Please select a valid continent.")

            # Convert numerical inputs
            beer = float(form_values["beer_servings"])
            spirit = float(form_values["spirit_servings"])
            wine = float(form_values["wine_servings"])

            values = [beer, spirit, wine]

            if not all(pd.notna(value) for value in values):
                raise ValueError("Enter valid numerical values.")

            if not all(value >= 0 for value in values):
                raise ValueError("Servings cannot be negative.")

            # Prepare categorical input using the saved imputer
            input_cat = pd.DataFrame([{
                "country": country,
                "continent": continent
            }])

            input_cat = pd.DataFrame(
                cat_imputer.transform(input_cat),
                columns=categorical_features
            )

            # Prepare numerical input using saved imputer and scaler
            input_num = pd.DataFrame([{
                "beer_servings": beer,
                "spirit_servings": spirit,
                "wine_servings": wine
            }])

            input_num = pd.DataFrame(
                num_imputer.transform(input_num),
                columns=numerical_features
            )

            # Encode categorical columns
            encoded_array = encoder.transform(input_cat)

            encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoded_columns
            )

            # Scale numerical columns
            scaled_array = scaler.transform(input_num)

            scaled_df = pd.DataFrame(
                scaled_array,
                columns=numerical_features
            )

            # Combine features in the same order as model training
            processed_input = pd.concat(
                [encoded_df, scaled_df],
                axis=1
            )

            # Predict
            result = model.predict(processed_input)[0]
            prediction = round(float(result), 2)

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid input values."

    return render_template(
        "index.html",
        countries=countries,
        continents=continents,
        prediction=prediction,
        error=error,
        form_values=form_values,
        model_name=metadata["model_name"]
    )


if __name__ == "__main__":
    app.run(debug=True)
