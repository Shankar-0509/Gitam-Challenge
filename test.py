import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[{
        'role': 'user',
        'content': '''Analyze this health report and return ONLY JSON:
{
    "overall_risk": "HIGH/MEDIUM/LOW",
    "detected_conditions": [],
    "major_risks": [],
    "minor_risks": [],
    "abnormal_values": [],
    "recommendations": [],
    "doctor_consultation": "YES/NO",
    "urgency": "IMMEDIATE/SOON/ROUTINE"
}

Patient Report:
Glucose: 180
Blood Pressure: 155
BMI: 32
Age: 45
Cholesterol: 250'''
    }]
)

print(response['message']['content'])