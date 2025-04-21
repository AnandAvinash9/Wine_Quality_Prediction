# Wine_Quality_Prediction
**Wine Quality Predictor App**    A **Streamlit** web app using **KNN** (70%+ accuracy) to classify wine quality into **Average/Premium/Exceptional**. Features: single prediction (sliders), batch CSV processing, probability visualization, and results export. For wineries, sommeliers, and QA teams. 

# 🍷 Wine Quality Prediction App

An interactive web application built with Streamlit to predict wine quality based on physicochemical properties using a trained K-Nearest Neighbors (KNN) classification model.

## 🚀 Features

- **Real-Time Prediction:** Adjust sliders for wine features and get instant predictions.
- **Batch Prediction:** Upload a CSV file to predict multiple wine samples at once.
- **Visualizations:** View prediction probabilities, data distributions, and summary statistics.

## 📊 Model Details

- **Model Type:** K-Nearest Neighbors (KNN)
- **Preprocessing:** Feature selection (`SelectKBest`), Standardization (`StandardScaler`)
- **Target Classes:** Wine quality mapped to classes like `Average`, `Premium`, `Exceptional`  
- **Libraries Used:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `streamlit`, `joblib`

## 🧪 Inputs

The model expects the following features:
- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`


## 🛠️ Installation

```bash
cd wine-quality-app
pip install -r requirements.txt
streamlit run wine_app.py

📁 Files
wine_app.py: Main Streamlit app

WineQT.csv: Dataset used for input reference and statistics

high_acc_wine_model.pkl: Pre-trained KNN model with pipeline

requirements.txt: Python dependencies

🌐 Live Demo 
deployed: https://winequalityprediction-6l3mc2mfck23jbj8nsxm7h.streamlit.app/
