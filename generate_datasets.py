"""
Synthetic Dataset Generator — Backup if Kaggle downloads fail
Run: python generate_datasets.py
"""
import pandas as pd
import numpy as np
import os

os.makedirs('datasets', exist_ok=True)
np.random.seed(42)
N = 800

print("Generating synthetic datasets...")

# ── 1. Diabetes Dataset ──────────────────────────────────────
n = N
glucose    = np.random.normal(120, 30, n).clip(60, 300)
bp         = np.random.normal(72, 12, n).clip(40, 120)
bmi        = np.random.normal(28, 7, n).clip(15, 55)
age        = np.random.randint(20, 80, n)
insulin    = np.random.normal(100, 80, n).clip(0, 500)
skin       = np.random.normal(25, 10, n).clip(5, 60)
pregnancies= np.random.randint(0, 15, n)
dpf        = np.random.uniform(0.1, 2.5, n)

# Label: diabetes if high glucose + high bmi + older age
risk = (glucose > 140).astype(int) + (bmi > 30).astype(int) + (age > 45).astype(int)
outcome = (risk >= 2).astype(int)
# Add some noise
flip = np.random.random(n) < 0.1
outcome = np.where(flip, 1 - outcome, outcome)

diabetes_df = pd.DataFrame({
    'Pregnancies': pregnancies,
    'Glucose': glucose.astype(int),
    'BloodPressure': bp.astype(int),
    'SkinThickness': skin.astype(int),
    'Insulin': insulin.astype(int),
    'BMI': np.round(bmi, 1),
    'DiabetesPedigreeFunction': np.round(dpf, 3),
    'Age': age,
    'Outcome': outcome
})
diabetes_df.to_csv('datasets/diabetes.csv', index=False)
print(f"✅ diabetes.csv — {len(diabetes_df)} rows | Positive: {outcome.sum()} ({outcome.mean()*100:.0f}%)")

# ── 2. Heart Disease Dataset ─────────────────────────────────
age2    = np.random.randint(30, 77, n)
sex     = np.random.randint(0, 2, n)
cp      = np.random.randint(0, 4, n)
trestbps= np.random.normal(130, 20, n).clip(90, 200).astype(int)
chol    = np.random.normal(245, 50, n).clip(120, 400).astype(int)
fbs     = (np.random.normal(110, 30, n) > 120).astype(int)
restecg = np.random.randint(0, 3, n)
thalach = np.random.normal(150, 22, n).clip(70, 210).astype(int)
exang   = np.random.randint(0, 2, n)
oldpeak = np.round(np.random.uniform(0, 5, n), 1)
slope   = np.random.randint(0, 3, n)
ca      = np.random.randint(0, 4, n)
thal    = np.random.randint(1, 4, n)

heart_risk = (trestbps > 140).astype(int) + (chol > 240).astype(int) + (age2 > 55).astype(int) + (oldpeak > 2).astype(int)
target = (heart_risk >= 2).astype(int)
flip2 = np.random.random(n) < 0.1
target = np.where(flip2, 1 - target, target)

heart_df = pd.DataFrame({
    'age': age2, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
    'thal': thal, 'target': target
})
heart_df.to_csv('datasets/heart.csv', index=False)
print(f"✅ heart.csv — {len(heart_df)} rows | Positive: {target.sum()} ({target.mean()*100:.0f}%)")

# ── 3. Kidney Disease Dataset ────────────────────────────────
bp3  = np.random.normal(76, 14, n).clip(50, 120).astype(int)
sg   = np.random.choice([1.005,1.010,1.015,1.020,1.025], n)
al   = np.random.choice([0,1,2,3,4,5], n, p=[0.5,0.2,0.1,0.1,0.05,0.05])
su   = np.random.choice([0,1,2,3,4,5], n, p=[0.6,0.15,0.1,0.08,0.04,0.03])
bgr  = np.random.normal(148, 70, n).clip(60, 400).astype(int)
bu   = np.random.normal(57, 40, n).clip(8, 200).astype(int)
sc   = np.round(np.random.exponential(1.5, n).clip(0.4, 15), 1)
hemo = np.round(np.random.normal(12.5, 2.5, n).clip(6, 17.5), 1)
pcv  = np.random.normal(38, 8, n).clip(20, 55).astype(int)
wbcc = np.random.normal(8400, 2800, n).clip(3000, 20000).astype(int)
rbcc = np.round(np.random.normal(4.5, 0.8, n).clip(2.0, 6.5), 1)

ckd_risk = (sc > 1.5).astype(int) + (bgr > 150).astype(int) + (al > 1).astype(int) + (hemo < 11).astype(int)
ckd_label = (ckd_risk >= 2).astype(int)
flip3 = np.random.random(n) < 0.08
ckd_label = np.where(flip3, 1 - ckd_label, ckd_label)
classification = np.where(ckd_label == 1, 'ckd', 'notckd')

kidney_df = pd.DataFrame({
    'bp': bp3, 'sg': sg, 'al': al, 'su': su, 'bgr': bgr,
    'bu': bu, 'sc': sc, 'hemo': hemo, 'pcv': pcv,
    'wbcc': wbcc, 'rbcc': rbcc, 'classification': classification
})
kidney_df.to_csv('datasets/kidney.csv', index=False)
print(f"✅ kidney.csv — {len(kidney_df)} rows | CKD: {ckd_label.sum()} ({ckd_label.mean()*100:.0f}%)")

# ── 4. Liver Disease Dataset ─────────────────────────────────
age4   = np.random.randint(20, 80, n)
gender = np.random.choice(['Male','Female'], n)
tb     = np.round(np.random.exponential(1.5, n).clip(0.3, 20), 1)
db     = np.round(tb * np.random.uniform(0.2, 0.8, n), 1)
alkphos= np.random.normal(240, 120, n).clip(60, 800).astype(int)
sgpt   = np.random.normal(60, 50, n).clip(10, 400).astype(int)
sgot   = np.random.normal(55, 45, n).clip(10, 400).astype(int)
tp     = np.round(np.random.normal(6.5, 1.0, n).clip(3, 9), 1)
alb    = np.round(np.random.normal(3.2, 0.7, n).clip(1.5, 5.5), 1)
ag     = np.round(alb / (tp - alb + 0.001), 2)

liver_risk = (tb > 2).astype(int) + (alkphos > 300).astype(int) + (sgpt > 100).astype(int) + (alb < 2.5).astype(int)
dataset = (liver_risk >= 2).astype(int) + 1  # 1=liver disease, 2=no disease
flip4 = np.random.random(n) < 0.1
dataset = np.where(flip4, 3 - dataset, dataset)

liver_df = pd.DataFrame({
    'Age': age4, 'Gender': gender,
    'Total_Bilirubin': tb, 'Direct_Bilirubin': db,
    'Alkaline_Phosphotase': alkphos,
    'Alamine_Aminotransferase': sgpt,
    'Aspartate_Aminotransferase': sgot,
    'Total_Protiens': tp, 'Albumin': alb,
    'Albumin_and_Globulin_Ratio': ag,
    'Dataset': dataset
})
liver_df.to_csv('datasets/liver.csv', index=False)
print(f"✅ liver.csv — {len(liver_df)} rows | Disease: {(dataset==1).sum()} ({(dataset==1).mean()*100:.0f}%)")

print("\n🎉 All 4 datasets generated in datasets/ folder!")
print("Now run: python classifier.py")
