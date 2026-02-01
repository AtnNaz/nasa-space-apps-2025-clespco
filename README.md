# A World Away: Hunting for Exoplanets with AI

## 🚀 Clesp Co.

### 🧑‍🤝‍🧑 Team

-   **Burak** – Machine Learning, Python, Web (HTML + CSS + JS)
-   **Ali** – Machine Learning, Python, Web Development
-   **Eren** – AI Engineering, Prompt Engineering

### 🛰️ Project Overview

This repository contains our submission for the **NASA Space Apps Challenge 2025**. Our project is an AI-driven pipeline designed to detect and classify exoplanets using real astronomical data from three major NASA missions. Within the 48-hour hackathon, we aimed to complete the full cycle of data collection, model training, and web demo development.

### 📊 Datasets Used

We utilized  publicly available datasets from three cornerstone NASA missions, which were processed and unified into a single machine learning-ready format:

-   **Kepler Objects of Interest (KOI)**
-   **TESS Objects of Interest (TOI)**
-   **K2 Planet Candidates (K2)**

### 🧪 Step-by-Step Technical Pipeline

Our entire technical process was implemented on the Kaggle platform.

**1. Library Imports**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings; warnings.filterwarnings('ignore')

path_koi = '/kaggle/input/datalar/koi.csv'
path_toi = '/kaggle/input/datalar/toi.csv'
path_k2 = '/kaggle/input/datalar/k2pandc.csv'

df_koi = pd.read_csv(path_koi, comment='#')
df_toi = pd.read_csv(path_toi, comment='#')
df_k2 = pd.read_csv(path_k2, comment='#')

print("verisetleri başarıyla yüklendi.")
```

**2. Data Loading & Unification**
- We loaded the three datasets and created a unified binary target column, is_planet, where 1 represents a confirmed planet or candidate and 0 represents a false positive.
```python
df_koi['is_planet'] = df_koi['koi_disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
df_koi.rename(columns={'kepler_name': 'planet_name'}, inplace=True)

df_toi['is_planet'] = df_toi['tfopwg_disp'].apply(lambda x: 1 if x in ['PC', 'KP'] else 0)
df_toi.rename(columns={'toi_id': 'planet_name'}, inplace=True)

df_k2['is_planet'] = df_k2['disposition'].apply(lambda x: 1 if x in ['CONFIRMED', 'CANDIDATE'] else 0)
df_k2.rename(columns={'pl_name': 'planet_name'}, inplace=True)

features_koi = {
    'planet_name': 'planet_name', 'koi_period': 'period', 'koi_duration': 'duration', 'koi_depth': 'depth',
    'koi_prad': 'planet_radius', 'koi_steff': 'stellar_temp', 'koi_srad': 'stellar_radius',
    'is_planet': 'is_planet'
}
df_koi_final = df_koi.rename(columns=features_koi)[list(features_koi.values())]

features_toi = {
    'planet_name': 'planet_name', 'Period (days)': 'period', 'Duration (hours)': 'duration', 'Depth (mmag)': 'depth',
    'Planet Radius (R_earth)': 'planet_radius', 'Stellar Eff Temp (K)': 'stellar_temp',
    'Stellar Radius (R_sun)': 'stellar_radius', 'is_planet': 'is_planet'
}
df_toi_final = df_toi.rename(columns=features_toi)[[col for col in features_toi.values() if col in df_toi.rename(columns=features_toi).columns]]

features_k2 = {
    'planet_name': 'planet_name', 'pl_orbper': 'period', 'pl_trandur': 'duration', 'pl_trandep': 'depth',
    'pl_rade': 'planet_radius', 'st_teff': 'stellar_temp', 'st_rad': 'stellar_radius',
    'is_planet': 'is_planet'
}
df_k2_final = df_k2.rename(columns=features_k2)[list(features_k2.values())]

df_final = pd.concat([df_koi_final, df_toi_final, df_k2_final], ignore_index=True)

df_final.dropna(inplace=True)

df_final = df_final[df_final['planet_name'].notna()]

df_final.to_csv('final_planet_dataset_with_names.csv', index=False)

print("birleştirilmiş ve temizlenmiş veriseti boyutu:", df_final.shape)
display(df_final.head())
```

**3. Feature Selection & Merging**
- We selected key physical features common across all missions (e.g., orbital period, planet radius, stellar temperature) and standardized their column names. The datasets were then merged and cleaned.
```python
X = df_final.drop(['is_planet', 'planet_name'], axis=1)
y = df_final['is_planet']
names = df_final['planet_name']

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, names, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

best_model_name = ""
best_model_instance = None
best_score = 0.0


for name, model in models.items():
    print(f"--- {name} Modeli Eğitiliyor ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Doğruluk (Accuracy): {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    if accuracy > best_score:
        best_score = accuracy
        best_model_name = name
        best_model_instance = model

print(f"\n{'='*30}")
print(f"En İyi Model: {best_model_name} (Doğruluk: {best_score:.4f})")
print(f"{'='*30}")
```

**4. Model Saving**
- The best-performing model was serialized and saved using joblib for deployment.
```python
model_filename = 'best_planet_classifier.joblib'
joblib.dump(best_model_instance, model_filename)

print(f"En iyi model olan '{best_model_name}', '{model_filename}' adıyla kaydedildi.")
```

**🌐 Deployment & Web Demo**
- We are developing a web interface for real-time classification. Users will be able to input transit data for a celestial object and receive a prediction from our model, indicating whether it's a potential exoplanet and with what confidence score.

## Frontend: HTML, CSS, JavaScript

## Backend (Planned): Flask API serving the planet_classifier.joblib model.

## Deployment: GitHub Pages + Render

**🛠️ Technologies**
- Data Science: Python, Pandas, Scikit-learn

- Machine Learning: LightGBM, XGBoost, RandomForest

- Platform: Kaggle Notebooks

- Web: Flask, HTML, CSS, JavaScript

## 🚀 Acknowledgements
- Thanks to NASA, Kaggle, and the Space Apps Challenge 2025 for making real-world science accessible and fostering global collaboration.

### 🛰️ Hackathon Experience

In NASA Space Apps 2025's 48-hour marathon:

1.  Data collection and preprocessing,
2.  Model design and training,
3.  Web demo development.

We will complete the steps.

## Katkıda Bulunanlar
- [@AtnNaz](https://github.com/AtnNaz)


### 📜 License

This project is licensed under the MIT License.
