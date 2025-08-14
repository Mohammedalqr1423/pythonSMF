from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model and encoders
bundle = joblib.load("stock_model_bundle.pkl")  # make sure this contains: {'model': ..., 'encoders': ...}
model = bundle['model']
label_encoders = bundle['encoders']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            'open', 'avg_22', 'close', 'num_deals',
            'sector', 'stock_code', 'stock_name'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        df = pd.DataFrame([{
            'إفتتاح': data['open'],
            'متوسط السعر آخر 22 جلسة': data['avg_22'],
            'إقفال': data['close'],
            'عدد الصفقات': data['num_deals'],
            'القطاع': data['sector'],
            'رمز الشركة': data['stock_code'],
            'اسم الشركة': data['stock_name'],
        }])

        for col in ['القطاع', 'رمز الشركة', 'اسم الشركة']:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        confidence = round(max(probabilities) * 100, 2)
        label = 'buy' if prediction == 1 else 'sell'

        return jsonify({
            'prediction': label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5000)
