import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs('models', exist_ok=True)
os.makedirs('datasets', exist_ok=True)

def print_header(title):
    print("\n" + "="*55)
    print(f"  {title}")
    print("="*55)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  💾 Saved → {path}")

# ── 1. DIABETES MODEL ────────────────────────────────────────
def train_diabetes():
    print_header("Training Model 1: Diabetes")
    try:
        df = pd.read_csv('datasets/diabetes.csv')
        print(f"  📂 Loaded {len(df)} rows")

        # Handle both column name variants
        col_map = {
            'bloodpressure': 'BloodPressure',
            'glucose': 'Glucose',
            'bmi': 'BMI',
            'age': 'Age',
            'insulin': 'Insulin',
            'outcome': 'Outcome'
        }
        df.columns = [col_map.get(c.lower(), c) for c in df.columns]

        features = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']
        # Use only available features
        features = [f for f in features if f in df.columns]
        
        X = df[features].fillna(df[features].median())
        y = df['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.1, random_state=42))
        ])
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        cv  = cross_val_score(model, X, y, cv=5).mean()
        print(f"  ✅ Accuracy:  {acc*100:.1f}%")
        print(f"  ✅ CV Score:  {cv*100:.1f}%")
        print(f"  ✅ Features:  {features}")

        save_model({'model': model, 'features': features}, 'models/diabetes.pkl')
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# ── 2. HEART DISEASE MODEL ───────────────────────────────────
def train_heart():
    print_header("Training Model 2: Heart Disease")
    try:
        df = pd.read_csv('datasets/heart.csv')
        print(f"  📂 Loaded {len(df)} rows")

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle target column variants
        if 'target' not in df.columns and 'condition' in df.columns:
            df['target'] = df['condition']
        if 'target' not in df.columns:
            # last column is usually target
            df = df.rename(columns={df.columns[-1]: 'target'})

        features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        features = [f for f in features if f in df.columns]

        X = df[features].fillna(df[features].median())
        y = df['target']
        # Binarize if needed
        if y.nunique() > 2:
            y = (y > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.1, random_state=42))
        ])
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        cv  = cross_val_score(model, X, y, cv=5).mean()
        print(f"  ✅ Accuracy:  {acc*100:.1f}%")
        print(f"  ✅ CV Score:  {cv*100:.1f}%")
        print(f"  ✅ Features:  {features}")

        save_model({'model': model, 'features': features}, 'models/heart.pkl')
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# ── 3. KIDNEY DISEASE MODEL ──────────────────────────────────
def train_kidney():
    print_header("Training Model 3: Kidney Disease (CKD)")
    try:
        df = pd.read_csv('datasets/kidney.csv')
        print(f"  📂 Loaded {len(df)} rows")

        df.columns = df.columns.str.lower().str.strip()
        df = df.fillna(df.mode().iloc[0])

        # Encode classification column
        if 'classification' in df.columns:
            df['classification'] = df['classification'].str.strip()
            le = LabelEncoder()
            df['classification'] = le.fit_transform(df['classification'])
            target_col = 'classification'
        else:
            target_col = df.columns[-1]

        features = ['bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc']
        features = [f for f in features if f in df.columns]

        X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = df[target_col].apply(pd.to_numeric, errors='coerce').fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=8,
                random_state=42, class_weight='balanced'))
        ])
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        cv  = cross_val_score(model, X, y, cv=5).mean()
        print(f"  ✅ Accuracy:  {acc*100:.1f}%")
        print(f"  ✅ CV Score:  {cv*100:.1f}%")
        print(f"  ✅ Features:  {features}")

        save_model({'model': model, 'features': features}, 'models/kidney.pkl')
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# ── 4. LIVER DISEASE MODEL ───────────────────────────────────
def train_liver():
    print_header("Training Model 4: Liver Disease")
    try:
        df = pd.read_csv('datasets/liver.csv')
        print(f"  📂 Loaded {len(df)} rows")

        df.columns = df.columns.str.strip()
        df = df.fillna(df.median(numeric_only=True))

        # Encode gender if present
        if 'Gender' in df.columns:
            df['Gender'] = (df['Gender'] == 'Male').astype(int)

        # Target column
        target_col = 'Dataset' if 'Dataset' in df.columns else df.columns[-1]
        y = df[target_col]
        y = (y == 1).astype(int)  # 1 = liver disease

        features = ['Total_Bilirubin', 'Alkaline_Phosphotase',
                    'Alamine_Aminotransferase', 'Albumin']
        features = [f for f in features if f in df.columns]
        if not features:
            # fallback to numeric columns
            features = df.select_dtypes(include=np.number).columns.tolist()
            features = [f for f in features if f != target_col][:6]

        X = df[features].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=150, max_depth=4,
                learning_rate=0.1, random_state=42))
        ])
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        cv  = cross_val_score(model, X, y, cv=5).mean()
        print(f"  ✅ Accuracy:  {acc*100:.1f}%")
        print(f"  ✅ CV Score:  {cv*100:.1f}%")
        print(f"  ✅ Features:  {features}")

        save_model({'model': model, 'features': features}, 'models/liver.pkl')
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

# ── PREDICT FUNCTION (used by app.py) ────────────────────────
def predict_all_diseases(data):
    """
    Takes a dict of patient values, returns disease predictions.
    data = {'Glucose': 180, 'BloodPressure': 120, 'BMI': 32, ...}
    """
    results = {}

    def safe_float(val, default=0):
        try:
            return float(val or default)
        except:
            return float(default)

    # Diabetes
    try:
        obj = pickle.load(open('models/diabetes.pkl', 'rb'))
        m, feats = obj['model'], obj['features']
        row = [[safe_float(data.get(f, data.get(f.lower(), 0))) for f in feats]]
        pred = m.predict(row)[0]
        prob = m.predict_proba(row)[0][1]
        results['Diabetes'] = {
            'status': 'Detected' if pred == 1 else 'Not Detected',
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"  Diabetes predict error: {e}")

    # Heart Disease
    try:
        obj = pickle.load(open('models/heart.pkl', 'rb'))
        m, feats = obj['model'], obj['features']
        feat_map = {
            'age': data.get('Age', data.get('age', 45)),
            'trestbps': data.get('Blood Pressure', data.get('trestbps', 120)),
            'chol': data.get('Cholesterol', data.get('chol', 200)),
            'thalach': data.get('Heart Rate', data.get('thalach', 150)),
            'oldpeak': data.get('oldpeak', 0)
        }
        row = [[safe_float(feat_map.get(f, 0)) for f in feats]]
        pred = m.predict(row)[0]
        prob = m.predict_proba(row)[0][1]
        results['Heart Disease'] = {
            'status': 'Detected' if pred == 1 else 'Not Detected',
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"  Heart predict error: {e}")

    # Kidney Disease
    try:
        obj = pickle.load(open('models/kidney.pkl', 'rb'))
        m, feats = obj['model'], obj['features']
        feat_map = {
            'bp': data.get('Blood Pressure', data.get('bp', 76)),
            'sg': data.get('sg', 1.015),
            'al': data.get('al', 0),
            'su': data.get('su', 0),
            'bgr': data.get('Glucose', data.get('bgr', 120)),
            'bu': data.get('bu', 40),
            'sc': data.get('Creatinine', data.get('sc', 1.0))
        }
        row = [[safe_float(feat_map.get(f, 0)) for f in feats]]
        pred = m.predict(row)[0]
        prob = m.predict_proba(row)[0][1]
        results['Kidney Disease (CKD)'] = {
            'status': 'Detected' if pred == 1 else 'Not Detected',
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"  Kidney predict error: {e}")

    # Liver Disease
    try:
        obj = pickle.load(open('models/liver.pkl', 'rb'))
        m, feats = obj['model'], obj['features']
        feat_map = {
            'Total_Bilirubin': data.get('Total_Bilirubin', 1.0),
            'Alkaline_Phosphotase': data.get('Alkaline_Phosphotase', 200),
            'Alamine_Aminotransferase': data.get('Alamine_Aminotransferase', 40),
            'Albumin': data.get('Albumin', 3.5)
        }
        row = [[safe_float(feat_map.get(f, 0)) for f in feats]]
        pred = m.predict(row)[0]
        prob = m.predict_proba(row)[0][1]
        results['Liver Disease'] = {
            'status': 'Detected' if pred == 1 else 'Not Detected',
            'confidence': round(prob * 100, 1)
        }
    except Exception as e:
        print(f"  Liver predict error: {e}")

    return results

# ── MAIN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "🏥 " * 18)
    print("  MediScan — Multi-Disease ML Training Pipeline")
    print("🏥 " * 18)

    results = {
        'Diabetes':      train_diabetes(),
        'Heart Disease': train_heart(),
        'Kidney Disease':train_kidney(),
        'Liver Disease': train_liver(),
    }

    print_header("Training Complete!")
    for disease, success in results.items():
        status = "✅ Ready" if success else "❌ Failed"
        print(f"  {status} — {disease}")

    trained = sum(results.values())
    print(f"\n  {trained}/4 models trained successfully")
    if trained > 0:
        print("  🚀 Run: python app.py")
    print("=" * 55)
