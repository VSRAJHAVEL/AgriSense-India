"""
Train Crop Recommendation Model using Random Forest Classifier.
Uses the generated Tamil Nadu crop dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train():
    csv_path = "data/Crop_recommendation.csv"
    if not os.path.exists(csv_path):
        print("Dataset not found! Run generate_crop_dataset.py first.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df['label'].nunique()} crops")
    
    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    feature_imp = dict(zip(X.columns, model.feature_importances_))
    print("Feature Importances:")
    for feat, imp in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le, "features": list(X.columns)}, "models/crop_recommender.pkl")
    print("\nModel saved to models/crop_recommender.pkl")

if __name__ == "__main__":
    train()
