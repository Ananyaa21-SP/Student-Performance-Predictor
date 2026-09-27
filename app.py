import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

if not os.path.exists("model/student_model.pkl"):
    st.error("Model not found. Please run train.py first.")
    st.stop()

data = pd.read_csv("dataset/StudentsPerformance.csv")
model = joblib.load("model/student_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/label_encoders.pkl")
features = joblib.load("model/feature_columns.pkl")
best_model_name = joblib.load("model/best_model_name.pkl")

st.sidebar.title("Student Performance Predictor")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Explore Data", "About"])

# HOME
if page == "Home":
    st.title("🎓 Student Performance Predictor")
    st.write("Can we estimate a student's Math Score from a few academic factors?")

    c1, c2, c3 = st.columns(3)
    c1.metric("Students", data.shape[0])
    c2.metric("Average Math Score", round(data.math_score.mean(), 1))
    c3.metric("Best Model", best_model_name)

    st.subheader("About the Project")
    st.write(
        "I chose this project to explore how machine learning can be used to "
        "estimate student performance. I kept the approach simple so that I "
        "could understand and explain each step of the ML process."
    )

    st.subheader("Project Flow")
    st.write("Data → Cleaning → EDA → Feature Selection → Model Training → Comparison → Prediction")

    st.subheader("Features Used for Prediction")
    st.write("Gender • Lunch Type • Test Preparation • Reading Score • Writing Score")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)

# PREDICTION
elif page == "Prediction":
    st.title("Predict Math Score")
    st.write("Enter the student's details and let the model estimate the Math Score.")

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", encoders["gender"].classes_)
        lunch = st.selectbox("Lunch Type", encoders["lunch"].classes_)
        test_prep = st.selectbox(
            "Test Preparation Course",
            encoders["test_preparation_course"].classes_
        )

    with c2:
        reading_score = st.slider("Reading Score", 0, 100, 70)
        writing_score = st.slider("Writing Score", 0, 100, 70)

    if st.button("Predict"):
        values = [
            encoders["gender"].transform([gender])[0],
            encoders["lunch"].transform([lunch])[0],
            encoders["test_preparation_course"].transform([test_prep])[0],
            reading_score,
            writing_score
        ]

        input_data = pd.DataFrame([values], columns=features)
        prediction = model.predict(scaler.transform(input_data))[0]
        prediction = float(np.clip(prediction, 0, 100))

        st.success(f"Predicted Math Score: {prediction:.2f} / 100")

        if prediction >= 80:
            band = "Excellent"
        elif prediction >= 60:
            band = "Good"
        elif prediction >= 40:
            band = "Average"
        else:
            band = "Needs Improvement"

        st.info(f"Performance Band: {band}")

# EXPLORE DATA
elif page == "Explore Data":
    st.title("Explore the Data")
    st.write("These visualizations helped me understand the dataset before training the models.")

    charts = [
        ("Score Distributions", "01_score_distributions.png",
         "Shows how Math, Reading and Writing scores are distributed."),
        ("Correlation Heatmap", "02_correlation_heatmap.png",
         "Shows how strongly the three subject scores are related."),
        ("Gender vs Performance", "03_gender_vs_performance.png",
         "Compares average scores between the gender groups."),
        ("Lunch Type vs Performance", "04_lunch_vs_performance.png",
         "Shows the difference in average scores between lunch groups."),
        ("Test Preparation vs Performance", "05_testprep_vs_performance.png",
         "Students who completed preparation generally have higher scores."),
        ("Pair Plot", "06_pairplot.png",
         "Shows relationships between the three subject scores."),
        ("Box Plots", "07_boxplots.png",
         "Shows score spread and helps identify unusual values.")
    ]

    for title, image, note in charts:
        st.subheader(title)
        st.image("images/" + image)
        st.caption(note)

    st.subheader("Model Comparison")
    comparison = pd.read_csv("model/model_comparison.csv")
    st.dataframe(comparison, use_container_width=True)
    st.image("images/08_model_comparison_r2.png")
    st.image("images/09_model_comparison_rmse.png")

    st.caption(
        f"{best_model_name} was selected because it achieved the highest R² score."
    )

    if os.path.exists("images/10_feature_importance.png"):
        st.subheader("Feature Importance")
        st.image("images/10_feature_importance.png")
    else:
        st.info("Feature importance is not available because the selected model is Linear Regression.")

# ABOUT
else:
    st.title("About This Project")

    st.subheader("My Approach")
    st.write(
        "I used three basic regression models and compared their performance "
        "using MAE, MSE, RMSE and R². The model with the highest R² was selected."
    )

    st.subheader("Model Results")
    comparison = pd.read_csv("model/model_comparison.csv")
    st.dataframe(comparison, use_container_width=True)
    st.success(f"Best Model: {best_model_name}")

    st.subheader("Algorithm Used")
    st.write(
        "Three models were trained and compared: Linear Regression, Decision Tree Regressor, and Random Forest Regressor. The best model based on R2 Score was chosen."
    )
    st.write("Best Model:", best_model_name)

    st.subheader("Limitations")
    st.write(
        "The prediction is an estimate based on this dataset and should not "
        "be treated as a definite measure of a student's actual performance."
    )

    st.subheader("Dataset")
    st.write(
        f"The dataset contains {data.shape[0]} student records and includes "
        "background information and Math, Reading and Writing scores."
    )

    st.subheader("Author")
    st.write("**Ananyaa S Pillai**")
    st.write("University Mini Project — Machine Learning")