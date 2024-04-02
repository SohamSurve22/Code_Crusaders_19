import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_road_width(api_key, latitude, longitude):
    # try:
        url = f"https://roads.googleapis.com/v1/nearestRoads?points={19.047321},{73.069908}&key={AIzaSyB8K55BHV2S-xvEOaZLatiCPowFoHcItHA}"
        response = requests.get(url)
        # image = cv2.imread('Photos/Map.jpg')
        if response.status_code == 200:
            data = response.json()
            if 'snappedPoints' in data and len(data['snappedPoints']) > 0:
                road_info = data['snappedPoints'][0]['roadInfo']
                if 'width' in road_info:
                    return road_info['width']
    except Exception as e:
        print("Error fetching road width:", e)
    return None

# Load parking data (replace this with your dataset)
parking_data = pd.read_csv("parking_data.csv")

# Extract features and target variable
X = parking_data.drop(columns=["availability"])
y = parking_data["availability"]

# Assume 'latitude' and 'longitude' are features representing location
# Add 'road_width' feature to X
X["road_width"] = X.apply(lambda row: get_road_width(api_key, row["latitude"], row["longitude"]), axis=1)

# Drop rows with missing road width data
X.dropna(subset=["road_width"], inplace=True)
y = y[X.index]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing pipeline (scaling)
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Define classifier pipeline (Random Forest)
classifier_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train classifier
classifier_pipeline.fit(X_train, y_train)

# Make predictions on test set
y_pred = classifier_pipeline.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))