# ==============================================================================
# IBM Applied Data Science Capstone - SpaceX Falcon 9 Landing Prediction
# Covers: API Collection, Wrangling, SQL EDA, Folium Map, Plotly Dash, & ML
# ==============================================================================

import requests
import pandas as pd
import numpy as np
import sqlite3
import folium
from folium.plugins import MarkerCluster
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------------------
# 1. DATA COLLECTION (SpaceX REST API) & DATA WRANGLING
# ------------------------------------------------------------------------------
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv"
df = pd.read_csv(url)

# Impute missing PayloadMass with mean
df['PayloadMass'] = df['PayloadMass'].fillna(df['PayloadMass'].mean())

# Create binary target label: 1 = Successful Landing, 0 = Failed / No Landing
landing_outcomes = df['Outcome'].value_counts()
bad_outcomes = {'False Ocean', 'False RTLS', 'False ASDS', 'None ASDS', 'None None'}
df['Class'] = df['Outcome'].apply(lambda x: 0 if x in bad_outcomes else 1)

print("Dataset Shape:", df.shape)
print("Overall Landing Success Rate:", round(df['Class'].mean() * 100, 2), "%")

# ------------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS WITH SQL (SQLite)
# ------------------------------------------------------------------------------
conn = sqlite3.connect(":memory:")
df.to_sql("SPACEXTBL", conn, if_exists="replace", index=False)

sql_queries = {
    "Unique Launch Sites": "SELECT DISTINCT LaunchSite FROM SPACEXTBL;",
    "Total Payload Mass": "SELECT SUM(PayloadMass) AS Total_Payload FROM SPACEXTBL;",
    "Success vs Failure Counts": "SELECT Class, COUNT(*) AS Count FROM SPACEXTBL GROUP BY Class;",
    "Success Rate by Launch Site": "SELECT LaunchSite, ROUND(AVG(Class)*100, 2) AS Success_Pct FROM SPACEXTBL GROUP BY LaunchSite;"
}

for title, query in sql_queries.items():
    print(f"\n--- {title} ---")
    print(pd.read_sql_query(query, conn))

# ------------------------------------------------------------------------------
# 3. INTERACTIVE VISUAL ANALYTICS WITH FOLIUM
# ------------------------------------------------------------------------------
launch_sites_df = df.groupby('LaunchSite', as_index=False).first()[['LaunchSite', 'Latitude', 'Longitude']]
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=4)
marker_cluster = MarkerCluster().add_to(site_map)

for _, row in df.iterrows():
    color = 'green' if row['Class'] == 1 else 'red'
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"Site: {row['LaunchSite']} | Outcome: {row['Outcome']}",
        icon=folium.Icon(color='white', icon_color=color)
    ).add_to(marker_cluster)

site_map.save("spacex_launch_sites_map.html")
print("\nSaved interactive Folium map to spacex_launch_sites_map.html")

# ------------------------------------------------------------------------------
# 4. PREDICTIVE ANALYSIS (CLASSIFICATION MODELS + GRIDSEARCHCV)
# ------------------------------------------------------------------------------
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights',
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

# One-hot encode categorical columns
X = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])
Y = df['Class'].to_numpy()

# Standardize features & split (80% train, 20% test)
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

models = {
    "Logistic Regression": (LogisticRegression(), {'C': [0.01, 0.1, 1], 'penalty': ['l2'], 'solver': ['lbfgs']}),
    "SVM": (SVC(), {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': np.logspace(-3, 3, 5), 'gamma': np.logspace(-3, 3, 5)}),
    "Decision Tree": (DecisionTreeClassifier(random_state=2), {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8], 'min_samples_split': [2, 5]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': list(range(1, 11)), 'algorithm': ['auto', 'ball_tree', 'kd_tree'], 'p': [1, 2]})
}

print("\n--- Model Evaluation Results ---")
for name, (estimator, params) in models.items():
    grid = GridSearchCV(estimator, params, cv=10)
    grid.fit(X_train, Y_train)
    test_acc = grid.score(X_test, Y_test)
    print(f"{name}: Best CV = {grid.best_score_:.4f} | Test Accuracy = {test_acc:.4f}")
    if name == "Decision Tree":
        yhat = grid.predict(X_test)
        print("Confusion Matrix:\n", confusion_matrix(Y_test, yhat))
