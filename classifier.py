import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train_all_models():
    train_diabetes()
    train_heart()
    train_kidney()
    train_liver()
    print("ALL MODELS TRAINED SUCCESSFULLY!")

def train_diabetes():
    df = pd.read_csv('datasets/diabetes.csv')
    X = df[['Glucose','BloodPressure',
            'BMI','Age','Insulin']]
    y = df['Outcome']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, 
        model.predict(X_test))
    print(f"Diabetes Model Accuracy: {acc*100:.1f}%")
    pickle.dump(model, 
        open('models/diabetes.pkl','wb'))

def train_heart():
    df = pd.read_csv('datasets/heart.csv')
    X = df[['age','trestbps','chol',
            'thalach','oldpeak']]
    y = df['target']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test,
        model.predict(X_test))
    print(f"Heart Model Accuracy: {acc*100:.1f}%")
    pickle.dump(model,
        open('models/heart.pkl','wb'))

def train_kidney():
    df = pd.read_csv('datasets/kidney.csv')
    df = df.fillna(df.mean())
    X = df[['bp','sg','al','su','bgr','bu','sc']]
    y = df['classification']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test,
        model.predict(X_test))
    print(f"Kidney Model Accuracy: {acc*100:.1f}%")
    pickle.dump(model,
        open('models/kidney.pkl','wb'))

def train_liver():
    df = pd.read_csv('datasets/liver.csv')
    df = df.fillna(df.mean())
    X = df[['Total_Bilirubin',
            'Alkaline_Phosphotase',
            'Alamine_Aminotransferase',
            'Albumin']]
    y = df['Dataset']
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test,
        model.predict(X_test))
    print(f"Liver Model Accuracy: {acc*100:.1f}%")
    pickle.dump(model,
        open('models/liver.pkl','wb'))

def predict_all_diseases(data):
    results = {}
    
    try:
        m = pickle.load(
            open('models/diabetes.pkl','rb'))
        pred = m.predict([[
            data.get('Glucose', 0),
            data.get('BloodPressure', 0),
            data.get('BMI', 0),
            data.get('Age', 0),
            data.get('Insulin', 0)
        ]])[0]
        results['Diabetes'] = \
            "Detected" if pred == 1 else "Not Detected"
    except Exception as e:
        print(f"Diabetes error: {e}")
    
    try:
        m = pickle.load(
            open('models/heart.pkl','rb'))
        pred = m.predict([[
            data.get('age', 0),
            data.get('trestbps', 0),
            data.get('chol', 0),
            data.get('thalach', 0),
            data.get('oldpeak', 0)
        ]])[0]
        results['Heart Disease'] = \
            "Detected" if pred == 1 else "Not Detected"
    except Exception as e:
        print(f"Heart error: {e}")
        
    return results