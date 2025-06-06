import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")


st.title("🍷 Wine Quality Prediction App")
st.markdown("""
This app predicts the **quality of wine** using a pre-trained K-Nearest Neighbors (KNN) model.
""")


@st.cache_data
def load_model():
    model_data = joblib.load('high_acc_wine_model.pkl')
    return model_data

model_data = load_model()
model = model_data['model']
selector = model_data['selector']
features = model_data['features']
quality_classes = model_data['classes']


scaler = model.named_steps['scaler']

@st.cache_data
def load_data():
    data = pd.read_csv('WineQT.csv')
    data = data.drop('Id', axis=1)
    return data

wine = load_data()
X = wine.drop('quality', axis=1)

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://github.com/AnandAvinash9/Wine_Quality_Prediction/blob/main/Augmented_Wine_Data.csv)
""")
st.sidebar.header("Model Information")
st.sidebar.markdown("""
- **Model Type:** Improved KNN Classifier  

- **Quality Classes:** Average, Premium, Exceptional
""")

if st.checkbox('Show raw data'):
    st.subheader('Wine Quality Dataset')
    st.write(wine)

if st.checkbox('Show dataset statistics'):
    st.subheader('Dataset Statistics')
    st.write(wine.describe())

st.header('Make Predictions')

input_method = st.radio("Select input method:", ("Sliders", "CSV Upload"))

if input_method == "Sliders":
    
    st.subheader('Single Prediction')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.slider('Fixed Acidity', float(X['fixed acidity'].min()), float(X['fixed acidity'].max()), float(X['fixed acidity'].mean()))
        volatile_acidity = st.slider('Volatile Acidity', float(X['volatile acidity'].min()), float(X['volatile acidity'].max()), float(X['volatile acidity'].mean()))
        citric_acid = st.slider('Citric Acid', float(X['citric acid'].min()), float(X['citric acid'].max()), float(X['citric acid'].mean()))
    
    with col2:
        residual_sugar = st.slider('Residual Sugar', float(X['residual sugar'].min()), float(X['residual sugar'].max()), float(X['residual sugar'].mean()))
        chlorides = st.slider('Chlorides', float(X['chlorides'].min()), float(X['chlorides'].max()), float(X['chlorides'].mean()))
        free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', float(X['free sulfur dioxide'].min()), float(X['free sulfur dioxide'].max()), float(X['free sulfur dioxide'].mean()))
    
    with col3:
        total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', float(X['total sulfur dioxide'].min()), float(X['total sulfur dioxide'].max()), float(X['total sulfur dioxide'].mean()))
        density = st.slider('Density', float(X['density'].min()), float(X['density'].max()), float(X['density'].mean()))
        pH = st.slider('pH', float(X['pH'].min()), float(X['pH'].max()), float(X['pH'].mean()))
    
    sulphates = st.slider('Sulphates', float(X['sulphates'].min()), float(X['sulphates'].max()), float(X['sulphates'].mean()))
    alcohol = st.slider('Alcohol', float(X['alcohol'].min()), float(X['alcohol'].max()), float(X['alcohol'].mean()))
    
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    
    st.subheader('Input Features')
    st.write(input_data)
    
    if st.button('Predict Quality'):
        try:

            X_selected = selector.transform(input_data)
            
            input_scaled = scaler.transform(X_selected)
            prediction = model.named_steps['knn'].predict(input_scaled)
            probability = model.named_steps['knn'].predict_proba(input_scaled)[0]  # Get first prediction's probabilities
            
            st.subheader('Prediction')
            st.write(f'Predicted Quality: **{prediction[0]}**')
            
           
            st.subheader('Prediction Probabilities')
            
            
            predicted_classes = model.named_steps['knn'].classes_
            
           
            prob_df = pd.DataFrame({
                'Quality Class': predicted_classes,
                'Probability': probability
            })
            
           
            st.bar_chart(prob_df.set_index('Quality Class'))
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

else:
    
    st.subheader('Batch Prediction via CSV Upload')
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            required_cols = X.columns.tolist()
            if not all(col in input_df.columns for col in required_cols):
                st.error(f"Error: The uploaded file must contain these columns: {', '.join(required_cols)}")
            else:
                st.subheader('Input Data')
                st.write(input_df)
                
                if st.button('Predict Quality for Batch'):
                    
                    X_selected = selector.transform(input_df[required_cols])
                   
                    input_scaled = scaler.transform(X_selected)
                    predictions = model.named_steps['knn'].predict(input_scaled)
                    probabilities = model.named_steps['knn'].predict_proba(input_scaled)
                    
                    
                    results_df = input_df.copy()
                    results_df['Predicted Quality'] = predictions
                    
                    
                    for i, cls in enumerate(model.named_steps['knn'].classes_):
                        results_df[f'P({cls})'] = probabilities[:, i]
                    
                    st.subheader('Prediction Results')
                    st.dataframe(results_df)
                    
                    
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='wine_quality_predictions.csv',
                        mime='text/csv'
                    )
                    
                    
                    st.subheader('Prediction Distribution')
                    fig, ax = plt.subplots()
                    sns.countplot(x='Predicted Quality', data=results_df, ax=ax)
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


st.header('Quality Distribution in Training Data')
fig, ax = plt.subplots()
sns.countplot(x='quality', data=wine, ax=ax)
st.pyplot(fig)

st.markdown("""
---
Avinash Anand🤖
""")
st.markdown("""
---
Avinash Anand
Created with ❤️ using Streamlit | Wine Quality Prediction App
""")
# Run The App 
# streamlit run wine_app.py 
