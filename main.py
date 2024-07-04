from flask import Flask, render_template, request, jsonify, render_template_string
import pandas as pd
import joblib  # Using joblib instead of pickle for loading the model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('final_delhi_dataset.csv')

# Load the trained model
model = joblib.load(open("model.pkl", 'rb'))

# Load the label encoders
le_dict = {}
cat_cols = data.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    col_safe = col.replace("/", "_")
    le_dict[col] = joblib.load(f'le_{col_safe}.pkl')

@app.route('/')
def index():
    bathrooms = sorted(data['numBathrooms'].unique())
    bedrooms = sorted(data['n_beds'].unique())
    BHK_RK = sorted(data['BHK_RK'].unique())
    buildingtype = sorted(data['building_type'].unique())
    status = sorted(data['Status'].unique())
 
 
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Housing Price Prediction</title>
    </head>
    <body>
        <h1>Enter Housing Details</h1>
        <form id="prediction-form">
            <label for="numBathrooms">Number of Bathrooms:</label>
            <select id="numBathrooms" name="numBathrooms">
                {% for bathroom in bathrooms %}
                    <option value="{{ bathroom }}">{{ bathroom }}</option>
                {% endfor %}
            </select><br>

            <label for="n_beds">Number of Bedrooms:</label>
            <select id="n_beds" name="n_beds">
                {% for bedroom in bedrooms %}
                    <option value="{{ bedroom }}">{{ bedroom }}</option>
                {% endfor %}
            </select><br>

            <label for="BHK/RK">BHK/RK:</label>
            <select id="BHK/RK" name="BHK/RK">
                {% for bhk in BHK_RK %}
                    <option value="{{ bhk }}">{{ bhk }}</option>
                {% endfor %}
            </select><br>

            <label for="building_type">Building Type:</label>
            <select id="building_type" name="building_type">
                {% for building in buildingtype %}
                    <option value="{{ building }}">{{ building }}</option>
                {% endfor %}
            </select><br>

            <label for="Status">Status:</label>
            <select id="Status" name="Status">
                {% for stat in status %}
                    <option value="{{ stat }}">{{ stat }}</option>
                {% endfor %}
            </select><br>

            <button type="submit">Predict Price</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('prediction-form').addEventListener('submit', function(event) {
                event.preventDefault();
                const formData = new FormData(event.target);
                fetch('/predict', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    document.getElementById('result').innerText = 'Predicted Price: ' + data.prediction;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
            });
        </script>
    </body>
    </html>
    ''', bathrooms=bathrooms, bedrooms=bedrooms, BHK_RK=BHK_RK, buildingtype=buildingtype, status=status)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    print(f"Input data: {input_data}")

    # Prepare the data for prediction
    features = {}
    for col in cat_cols:
        col_safe = col.replace("/", "_")  # Adjust for '/' replaced with '_'
        try:
            print(f"Accessing col_safe: {col_safe}")
            features[col] = le_dict[col_safe].transform([input_data[col]])[0]
        except KeyError as e:
            print(f"KeyError: {e} - col: {col}, input_data[col]: {input_data.get(col)}")
            return jsonify({'error': f"Invalid input: {col}"})

    df_features = pd.DataFrame(features, index=[0])

    # Make prediction
    prediction = model.predict(df_features)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)