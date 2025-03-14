import pickle
from flask import Flask, request, jsonify
from statsmodels.tsa.statespace.sarimax import SARIMAX


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Flask App is Running!</p>"


model_pickle = open('model.pkl', 'rb')
model = pickle.load(model_pickle)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json

        # Extract required fields
        page = data['page']
        start_date = data['start_date']
        end_date = data['end_date']
        campaign = data['campaign']

        # Make prediction using SARIMAX model
        # Adjust based on your model's requirements
        prediction = model.predict(
            start=start_date,
            end=end_date,
            exog=campaign if data['language'] == 'en' else None
        )

        # Format response
        response = {
            'predictions': [
                {
                    'date': str(date),
                    'predicted_views': float(value)
                } for date, value in zip(prediction.index, prediction.values)
            ],
            'metadata': {
                'page': page,
                'start_date': start_date,
                'end_date': end_date
            }
        }

        return jsonify(response)

    except KeyError as e:
        return jsonify({'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask App
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)