"""
predict.py
==========
Student Performance Predictor - Command Line Prediction Script

This script loads the trained model (saved by train.py) and asks the user
for a few details in the terminal, then predicts the student's Math Score.

Run with:  python predict.py
(Make sure you have already run train.py at least once so the 'model/'
 folder contains the saved .pkl files.)
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_artifacts():
    """Load the trained model and all preprocessing objects from disk."""
    model = joblib.load(os.path.join(MODEL_DIR, "student_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    best_model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    return model, scaler, label_encoders, feature_columns, best_model_name


def get_choice(prompt, options):
    """Ask the user to pick from a small list of valid options (case-insensitive)."""
    options_lower = [o.lower() for o in options]
    while True:
        value = input(f"{prompt} ({'/'.join(options)}): ").strip().lower()
        if value in options_lower:
            return options[options_lower.index(value)]
        print(f"Invalid input. Please choose one of: {', '.join(options)}")


def get_score(prompt):
    """Ask the user for a score between 0 and 100."""
    while True:
        try:
            value = float(input(f"{prompt} (0-100): ").strip())
            if 0 <= value <= 100:
                return value
            print("Score must be between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")


def predict_math_score(model, scaler, label_encoders, feature_columns,
                        gender, lunch, test_prep, reading_score, writing_score):
    """
    Build a single-row feature vector in the exact order the model expects,
    encode the categorical values using the SAME encoders used in training,
    scale it using the SAME scaler used in training, then predict.
    """
    gender_encoded = label_encoders["gender"].transform([gender])[0]
    lunch_encoded = label_encoders["lunch"].transform([lunch])[0]
    test_prep_encoded = label_encoders["test_preparation_course"].transform([test_prep])[0]

    # Must match the order of feature_columns used during training:
    # ["gender", "lunch", "test_preparation_course", "reading_score", "writing_score"]
    import pandas as pd
    input_data = pd.DataFrame(
        [[gender_encoded, lunch_encoded, test_prep_encoded, reading_score, writing_score]],
        columns=feature_columns,
    )

    input_scaled = scaler.transform(input_data)
    predicted_math_score = model.predict(input_scaled)[0]

    # Clip prediction to a realistic 0-100 range
    predicted_math_score = float(np.clip(predicted_math_score, 0, 100))
    return round(predicted_math_score, 2)


def main():
    print("=" * 55)
    print("   STUDENT PERFORMANCE PREDICTOR - Math Score")
    print("=" * 55)

    model, scaler, label_encoders, feature_columns, best_model_name = load_artifacts()
    print(f"Loaded model: {best_model_name}\n")

    gender = get_choice("Enter Gender", ["male", "female"])
    lunch = get_choice("Enter Lunch Type", ["standard", "free/reduced"])
    test_prep = get_choice("Test Preparation Course", ["completed", "none"])
    reading_score = get_score("Enter Reading Score")
    writing_score = get_score("Enter Writing Score")

    predicted_score = predict_math_score(
        model, scaler, label_encoders, feature_columns,
        gender, lunch, test_prep, reading_score, writing_score
    )

    print("\n" + "-" * 55)
    print(f"Predicted Math Score: {predicted_score} / 100")
    print("-" * 55)


if __name__ == "__main__":
    main()
