from flask import Flask, request, jsonify, render_template
import pandas as pd
import pdfplumber
import pickle
import numpy as np
import ollama

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['report']
    
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        text = df.to_string()
        data = df.to_dict()
    
    elif file.filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        data = {"raw_text": text}
    
    # Run classifiers
    results = predict_all_diseases(data)
    
    # Run Ollama analysis
    ollama_result = analyze_with_ollama(text)
    
    return jsonify({
        "classified": results,
        "full_analysis": ollama_result
    })

if __name__ == '__main__':
    app.run(debug=True)