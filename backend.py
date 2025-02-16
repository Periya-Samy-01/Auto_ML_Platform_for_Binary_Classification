import os
import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict
import json
import joblib
from datetime import datetime

app = FastAPI()
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def train_model(X_train, y_train, X_test, y_test, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_pred)
    }
    return best_model, results

@app.post("/upload/")
async def upload_dataset(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}

@app.post("/train/")
async def train_models(filename: str = Form(...), models: List[str] = Form(...)):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_mappings = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
        "HistGradientBoosting": HistGradientBoostingClassifier()
    }

    results_dict = {}

    def train_single_model(model_name):
        model = model_mappings[model_name]
        trained_model, metrics = train_model(X_train, y_train, X_test, y_test, model, {})
        results_dict[model_name] = metrics  # ✅ Store results correctly

        # ✅ Save model with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(MODEL_FOLDER, model_filename)

        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)
        
        return model_name, metrics, model_filename

    trained_models = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(train_single_model)(model_name) for model_name in models
    )

    # ✅ Convert results to correct format
    results_dict = {model_name: metrics for model_name, metrics, _ in trained_models}

    # ✅ Automated Best Model Selection
    best_model = max(trained_models, key=lambda x: x[1]["accuracy"])
    best_model_name, best_model_metrics, best_model_file = best_model

    return {"results": results_dict, "best_model": best_model_file}


@app.get("/get_models/")
def get_models():
    models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    return {"models": models}

@app.get("/download/{model_name}")
def download_model(model_name: str):
    file_path = os.path.join(MODEL_FOLDER, model_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=model_name)
    return {"error": "Model not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
