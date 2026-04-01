import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# --- Random Forest Model ---
def train_baseline_model():
    """
    Loads features, combines them, and trains Random Forest 
    and saves the trained model to disk.
    """
    print("Loading feature datasets...")
    features_path = 'data/processed/clean_features.csv'
    fft_path = 'data/processed/fft_features.csv'
    
    features_df = pd.read_csv(features_path)
    fft_df = pd.read_csv(fft_path)
    
    print("Merging datasets (Data Fusion)...")
    df_merged = pd.merge(features_df, fft_df, on=['id', 'class'])
    
    X = df_merged.drop(columns=['id', 'class'])
    y = df_merged['class']
    
    print("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Initializing Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Training model... (This may take a minute)")
    model.fit(X_train, y_train)
    
    print("\nEvaluating model on Test set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Check a models folder for it does exist
    os.makedirs('models', exist_ok=True)
    
    model_save_path = 'models/rf_baseline.pkl'
    print(f"Serializing and saving model to {model_save_path}...")
    joblib.dump(model, model_save_path)
    
    print("Training Pipeline Complete.")


















if __name__ == "__main__":
    train_baseline_model()