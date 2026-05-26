# -----------------------------------
# IMPORT LIBRARIES
# -----------------------------------
from flask import send_file
from fpdf import FPDF
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle


# -----------------------------------
# LOAD DATASET
# -----------------------------------

data = pd.read_csv("house_data (3).csv")


# -----------------------------------
# DATA CLEANING
# -----------------------------------

# Remove unwanted columns
data = data.drop(['area_type', 'availability', 'society'], axis=1)

# Remove missing values
data = data.dropna()


# -----------------------------------
# CONVERT SIZE COLUMN
# Example:
# 2 BHK -> 2
# 4 Bedroom -> 4
# -----------------------------------

data['size'] = data['size'].str.extract('(\d+)')

data['size'] = data['size'].astype(int)


# -----------------------------------
# CONVERT total_sqft COLUMN
# -----------------------------------

data['total_sqft'] = pd.to_numeric(
    data['total_sqft'],
    errors='coerce'
)

# Remove invalid rows
data = data.dropna()


# -----------------------------------
# ENCODE LOCATION COLUMN
# -----------------------------------

le = LabelEncoder()

data['location'] = le.fit_transform(data['location'])


# -----------------------------------
# FEATURES & TARGET
# -----------------------------------

X = data[['location', 'size', 'total_sqft', 'bath', 'balcony']]

y = data['price']


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# TRAIN MODEL
# -----------------------------------

model = RandomForestRegressor()

model.fit(X_train, y_train)
# Model Accuracy

accuracy = model.score(X_test, y_test)

accuracy = round(accuracy * 100, 2)

# Total houses

total_houses = len(data)


# -----------------------------------
# SAVE MODEL
# -----------------------------------

pickle.dump(model, open("model.pkl", "wb"))


# -----------------------------------
# CREATE FLASK APP
# -----------------------------------

app = Flask(__name__)


# Load model
model = pickle.load(open("model.pkl", "rb"))


# -----------------------------------
# HOME PAGE
# -----------------------------------
latest_prediction = {}
@app.route('/')
def home():

    return render_template(
        'index.html',
        accuracy=accuracy,
        total_houses=total_houses
    )

# -----------------------------------
# PREDICTION ROUTE
# -----------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    global latest_prediction

    # Get form values
    location = request.form['location']

    size = int(request.form['size'])

    total_sqft = float(request.form['total_sqft'])

    bath = int(request.form['bath'])

    balcony = int(request.form['balcony'])


    # Encode location
    location_encoded = le.transform([location])[0]


    # Create feature array
    features = np.array([
        [location_encoded, size, total_sqft, bath, balcony]
    ])


    # Predict price
    prediction = model.predict(features)


    # Round output
    output = round(prediction[0], 2)


    # Store latest prediction
    latest_prediction = {
        "location": location,
        "size": size,
        "sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "price": output
    }


    # Return result
    return render_template(
        'index.html',
        prediction_text=f'Estimated House Price: ₹ {output} Lakhs',
        accuracy=accuracy,
        total_houses=total_houses
    )

  

# -----------------------------------
# RUN APP
# -----------------------------------
@app.route('/download_pdf')

def download_pdf():

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="House Price Prediction Report",
             ln=True, align='C')

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10,
             txt=f"Location: {latest_prediction['location']}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"BHK Size: {latest_prediction['size']}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Total Sqft: {latest_prediction['sqft']}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Bathrooms: {latest_prediction['bath']}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Balcony: {latest_prediction['balcony']}",
             ln=True)

    pdf.cell(200, 10,
             txt=f"Predicted Price: Rs {latest_prediction['price']} Lakhs",
             ln=True)

    pdf.output("report.pdf")

    return send_file("report.pdf", as_attachment=True)
if __name__ == "__main__":
    app.run(debug=True)