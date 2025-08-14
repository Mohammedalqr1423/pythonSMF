from flask import Flask, request, jsonify
import pandas as pd
import joblib
import gdown
from pathlib import Path

app = Flask(__name__)

# --------------------
# تحميل الموديل إذا لم يكن موجود
# --------------------
def download_model_from_gdrive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(destination), quiet=False)

model_path = Path("stock_model_bundle.pkl")
if not model_path.exists():
    print("📥 جاري تحميل الموديل...")
    gdrive_file_id = "173L7guUnDsSrqi3X1t6z3iT110rSA_t7"  # File ID من Google Drive
    download_model_from_gdrive(gdrive_file_id, model_path)

# --------------------
# تحميل الموديل والـ Encoders
# --------------------
bundle = joblib.load(model_path)
model = bundle['model']
label_encoders = bundle['encoders']

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # الحقول المطلوبة
        required_fields = [
            'open', 'avg_22', 'close', 'num_deals',
            'sector', 'stock_code', 'stock_name'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # تجهيز البيانات بنفس أسماء الأعمدة في التدريب
        df = pd.DataFrame([{
            'إفتتاح': data['open'],
            'متوسط السعر آخر 22 جلسة': data['avg_22'],
            'إقفال': data['close'],
            'عدد الصفقات': data['num_deals'],
            'القطاع': data['sector'],
            'رمز الشركة': data['stock_code'],
            'اسم الشركة': data['stock_name'],
        }])

        # تحويل القيم النصية إلى أرقام باستخدام الـ Encoders
        for col in ['القطاع', 'رمز الشركة', 'اسم الشركة']:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

        # عمل التنبؤ
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
