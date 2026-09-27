import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("model", exist_ok=True)
os.makedirs("images", exist_ok=True)
sns.set_style("whitegrid")

# 1. Load and clean data
data = pd.read_csv("dataset/StudentsPerformance.csv")
print("Dataset shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())

scores = ["math_score", "reading_score", "writing_score"]
data[scores] = data[scores].fillna(data[scores].median())
data = data.drop_duplicates()

# 2. Exploratory Data Analysis

# Score distributions
plt.figure(figsize=(15, 4))
for i, col in enumerate(scores):
    plt.subplot(1, 3, i + 1)
    sns.histplot(data[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.savefig("images/01_score_distributions.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(data[scores].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/02_correlation_heatmap.png")
plt.close()

# Gender, lunch and test preparation
groups = [
    ("gender", "03_gender_vs_performance.png", "Average Scores by Gender"),
    ("lunch", "04_lunch_vs_performance.png", "Average Scores by Lunch Type"),
    ("test_preparation_course", "05_testprep_vs_performance.png", "Average Scores by Test Preparation")
]

for column, filename, title in groups:
    data.groupby(column)[scores].mean().plot(kind="bar", figsize=(7, 5))
    plt.title(title)
    plt.ylabel("Average Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("images/" + filename)
    plt.close()

# Pair plot
plot = sns.pairplot(data[scores + ["gender"]], hue="gender")
plot.savefig("images/06_pairplot.png")
plt.close()

# Box plots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(data=data, x="test_preparation_course", y="math_score", ax=ax[0])
sns.boxplot(data=data, x="lunch", y="reading_score", ax=ax[1])
sns.boxplot(data=data, x="gender", y="writing_score", ax=ax[2])
ax[0].set_title("Math Score vs Test Preparation")
ax[1].set_title("Reading Score vs Lunch")
ax[2].set_title("Writing Score vs Gender")
plt.tight_layout()
plt.savefig("images/07_boxplots.png")
plt.close()

# 3. Encode categorical features
categorical = ["gender", "lunch", "test_preparation_course"]
encoders = {}

for col in categorical:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])

# 4. Select features and target
features = [
    "gender", "lunch", "test_preparation_course",
    "reading_score", "writing_score"
]
X, y = data[features], data["math_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42
    )
}

trained = {}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    trained[name] = model
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, pred), 3),
        "MSE": round(mse, 3),
        "RMSE": round(np.sqrt(mse), 3),
        "R2 Score": round(r2_score(y_test, pred), 3)
    })

# 6. Compare models
results = pd.DataFrame(results).sort_values(
    "R2 Score", ascending=False
).reset_index(drop=True)

print("\nModel Comparison:\n", results)
results.to_csv("model/model_comparison.csv", index=False)

# R2 comparison
sns.barplot(data=results, x="Model", y="R2 Score")
plt.title("Model Comparison - R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/08_model_comparison_r2.png")
plt.close()

# RMSE comparison
sns.barplot(data=results, x="Model", y="RMSE")
plt.title("Model Comparison - RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("images/09_model_comparison_rmse.png")
plt.close()

# 7. Select best model
best_name = results.iloc[0]["Model"]
best_model = trained[best_name]
print("\nBest Model:", best_name)

# Feature importance for tree-based best model
if best_name in ["Decision Tree Regressor", "Random Forest Regressor"]:
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    sns.barplot(data=importance, x="Importance", y="Feature")
    plt.title("Feature Importance - " + best_name)
    plt.tight_layout()
    plt.savefig("images/10_feature_importance.png")
    plt.close()
    importance.to_csv("model/feature_importance.csv", index=False)

# 8. Save model and preprocessing objects
joblib.dump(best_model, "model/student_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/label_encoders.pkl")
joblib.dump(features, "model/feature_columns.pkl")
joblib.dump(best_name, "model/best_model_name.pkl")

print("\nModel and supporting files saved successfully!")
print("Training complete!")
