from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pdfplumber
import numpy as np
import ollama
import json
import re
import os

from classifier import predict_all_diseases

app = Flask(__name__)
CORS(app)
os.makedirs('uploads', exist_ok=True)

def analyze_with_ollama(health_text, diseases, risk_level):
    detected = [d for d, v in diseases.items() if v['status'] == 'Detected']
    disease_str = ", ".join(detected) if detected else "None detected"
    confidence_str = "\n".join([
        f"- {d}: {v['status']} ({v['confidence']}% confidence)"
        for d, v in diseases.items()
    ])
    prompt = f"""You are a senior clinical physician. The ML models have analyzed this patient and produced these results:

ML MODEL PREDICTIONS:
{confidence_str}
Overall Risk: {risk_level}

PATIENT DATA:
{health_text}

Return ONLY this JSON (no markdown, no extra text):
{{
    "overall_risk": "{risk_level}",
    "detected_conditions": {json.dumps(detected)},
    "major_risks": ["serious risks with medical reasoning"],
    "minor_risks": ["moderate risks"],
    "abnormal_values": ["Value: X (normal: Y-Z) — clinical meaning"],
    "recommendations": ["specific actionable medical steps"],
    "doctor_consultation": "YES or NO",
    "urgency": "IMMEDIATE or SOON or ROUTINE",
    "summary": "2-3 sentence clinical summary mentioning the ML-detected diseases"
}}"""
    try:
        response = ollama.chat(
            model='llama3.2:latest',
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = response.message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        return json.loads(match.group() if match else raw)
    except Exception as e:
        print(f"Ollama error: {e}")
        return {
            "overall_risk": risk_level,
            "detected_conditions": detected,
            "major_risks": [f"{d} detected by ML model" for d in detected],
            "minor_risks": [],
            "abnormal_values": [],
            "recommendations": ["Please consult a doctor for full evaluation"],
            "doctor_consultation": "YES",
            "urgency": "SOON" if risk_level == "HIGH" else "ROUTINE",
            "summary": f"ML analysis detected: {disease_str}. Risk level: {risk_level}."
        }

def calculate_risk(diseases):
    detected = sum(1 for v in diseases.values() if v['status'] == 'Detected')
    high_conf = sum(1 for v in diseases.values() if v['status'] == 'Detected' and v['confidence'] > 75)
    if detected >= 3 or high_conf >= 2:
        return "HIGH"
    elif detected >= 1:
        return "MEDIUM"
    return "LOW"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze/manual', methods=['POST'])
def analyze_manual():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No data received"}), 400
        diseases = predict_all_diseases(data)
        risk_level = calculate_risk(diseases)
        health_text = "\n".join([f"{k}: {v}" for k, v in data.items() if v])
        result = analyze_with_ollama(health_text, diseases, risk_level)
        result['ml_predictions'] = diseases
        result['detected_conditions'] = [
            f"{d} ({v['confidence']}% confidence)"
            for d, v in diseases.items() if v['status'] == 'Detected'
        ] or result.get('detected_conditions', [])
        return jsonify({"success": True, "result": result})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/analyze/pdf', methods=['POST'])
def analyze_pdf():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files['file']
        if not file.filename.endswith('.pdf'):
            return jsonify({"success": False, "error": "Only PDF files allowed"}), 400
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        os.remove(filepath)
        if not text.strip():
            return jsonify({"success": False, "error": "Could not extract text from PDF"}), 400
        parsed = {}
        patterns = {
            'Glucose': r'[Gg]lucose[:\s]+(\d+\.?\d*)',
            'Blood Pressure': r'[Ss]ystolic[:\s]+(\d+)|[Bb]lood\s*[Pp]ressure[:\s]+(\d+)',
            'BMI': r'BMI[:\s]+(\d+\.?\d*)',
            'Cholesterol': r'[Cc]holesterol[:\s]+(\d+\.?\d*)',
            'Creatinine': r'[Cc]reatinine[:\s]+(\d+\.?\d*)',
            'Age': r'[Aa]ge[:\s]+(\d+)',
            'Heart Rate': r'[Hh]eart\s*[Rr]ate[:\s]+(\d+)',
            'HbA1c': r'HbA1c[:\s]+(\d+\.?\d*)',
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text)
            if m:
                parsed[key] = m.group(1) or (m.group(2) if m.lastindex and m.lastindex >= 2 else None)
        diseases = predict_all_diseases(parsed)
        risk_level = calculate_risk(diseases)
        result = analyze_with_ollama(text[:2500], diseases, risk_level)
        result['ml_predictions'] = diseases
        result['detected_conditions'] = [
            f"{d} ({v['confidence']}% confidence)"
            for d, v in diseases.items() if v['status'] == 'Detected'
        ] or result.get('detected_conditions', [])
        return jsonify({"success": True, "result": result})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/analyze/csv', methods=['POST'])
def analyze_csv():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files['file']
        df = pd.read_csv(file)
        data = df.iloc[0].to_dict()
        diseases = predict_all_diseases(data)
        risk_level = calculate_risk(diseases)
        health_text = "\n".join([f"{k}: {v}" for k, v in data.items()])
        result = analyze_with_ollama(health_text, diseases, risk_level)
        result['ml_predictions'] = diseases
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    models_status = {name: os.path.exists(f'models/{name}.pkl')
                     for name in ['diabetes', 'heart', 'kidney', 'liver']}
    return jsonify({
        "models": models_status,
        "diseases_covered": ["Diabetes", "Heart Disease", "Kidney Disease (CKD)", "Liver Disease"],
        "ml_algorithms": {
            "diabetes": "Gradient Boosting Classifier",
            "heart": "Gradient Boosting Classifier",
            "kidney": "Random Forest Classifier",
            "liver": "Gradient Boosting Classifier"
        },
        "llm": "Llama 3.2 (local via Ollama)",
        "datasets": {
            "diabetes": "Pima Indians Diabetes Dataset (768 patients)",
            "heart": "Cleveland Heart Disease UCI (303 patients)",
            "kidney": "UCI CKD Dataset (400 patients)",
            "liver": "Indian Liver Patient Dataset (583 patients)"
        }
    })

if __name__ == '__main__':
    print("\n🏥 MediScan — Health Risk Early Detection Dashboard")
    print("=" * 50)
    for name in ['diabetes', 'heart', 'kidney', 'liver']:
        status = "✅" if os.path.exists(f'models/{name}.pkl') else "❌ run classifier.py first"
        print(f"  {name.capitalize()} model: {status}")
    print("=" * 50)
    print("  Running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
