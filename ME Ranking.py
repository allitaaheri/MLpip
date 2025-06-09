import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import requests

# Full raw URL to the file in your GitHub repository
url = "https://github.com/allitaaheri/MLpip/raw/main/ME%20%26%20IE%20Department%20Data.xlsx"

# Output path (you can rename the file as needed)
output_path = "ME & IE Department Data.xlsx"

response = requests.get(url)
if response.status_code == 200:
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"File downloaded to {output_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

df = pd.read_excel(output_path, engine="openpyxl", sheet_name='Mechanical Engineering Summary ')

columns_map = {
    'University': 'University',
    '# of Faculty (5/12/23)': 'Number of Faculty',
    '# of Tenure (5/12/23)': 'Number of Tenure',
    'P1': 'P1',
    'ENROLLMENT (FULL-TIME) - 2023': 'Enrollment',
    'E2': 'E2',
    'Total funds - 2021': 'Total funds 2021',
    'Total funds - 2022': 'Total funds 2022',
    'AVERAGE QUANTITATIVE GRE - 2023': 'GRE Score',
    'ME Ranking - 2023': 'ME Ranking 2023',
    'ME Ranking - 2024': 'ME Ranking 2024'
}
data = df[list(columns_map.keys())].copy()
data.columns = list(columns_map.values())


def extract_rank(rank_str):
    if pd.isna(rank_str):
        return np.nan
    if isinstance(rank_str, str) and '#' in rank_str:
        match = re.search(r'#(\d+)', rank_str)
        if match:
            return int(match.group(1))
    try:
        return float(rank_str)
    except:
        return np.nan
data['ME Ranking 2023'] = data['ME Ranking 2023'].apply(extract_rank)
data['ME Ranking 2024'] = data['ME Ranking 2024'].apply(extract_rank)

data = data.dropna(subset=['ME Ranking 2023', 'ME Ranking 2024'])
data['Label'] = (data['ME Ranking 2024'] < data['ME Ranking 2023']).astype(int)

numerical_features = ['Number of Faculty', 'Number of Tenure', 'P1', 'Enrollment',
                      'E2', 'Total funds 2021', 'Total funds 2022', 'GRE Score', 'ME Ranking 2023']
for col in numerical_features:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].fillna(data[col].mean())

from sklearn.cluster import KMeans

# KMeans-based Imputation for Invalid Funding 
bad_2021 = data['Total funds 2021'] <= 0
bad_2022 = data['Total funds 2022'] <= 0


valid_rows = ~(bad_2021 | bad_2022)
features_for_kmeans = data.loc[valid_rows, numerical_features]
kmeans = KMeans(n_clusters=3, random_state=12)
kmeans.fit(features_for_kmeans)

data['Cluster'] = kmeans.predict(data[numerical_features])

# Impute based on cluster
for cluster_id in np.unique(data['Cluster']):
    cluster_mask = data['Cluster'] == cluster_id

    # Impute Total funds 2021
    mean_2021 = data.loc[cluster_mask & ~bad_2021, 'Total funds 2021'].mean()
    data.loc[cluster_mask & bad_2021, 'Total funds 2021'] = mean_2021

    # Impute Total funds 2022
    mean_2022 = data.loc[cluster_mask & ~bad_2022, 'Total funds 2022'].mean()
    data.loc[cluster_mask & bad_2022, 'Total funds 2022'] = mean_2022

# Step 4: Drop cluster label
data = data.drop(columns=['Cluster'])

log_transform_cols = ['Enrollment', 'Total funds 2021', 'Total funds 2022']

for col in log_transform_cols:
    if (data[col] <= 0).any():
        print(f"Skipping log transform for {col} — contains non-positive values.")
    else:
        data[col] = np.log1p(data[col])
        print(f"Log-transformed: {col}")

from sklearn.metrics import roc_curve

# Z-score threshold for outlier removal
threshold_zscore = 3

X = data[numerical_features]
y = data['Label']

from scipy.stats import zscore

z_scores = np.abs(zscore(X))
non_outliers = (z_scores < threshold_zscore).all(axis=1)

# Apply filter
X = X[non_outliers]
y = y[non_outliers]

print(f"Removed {len(z_scores) - non_outliers.sum()} outliers. Remaining samples: {non_outliers.sum()}")
poly = PolynomialFeatures(degree=40, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(numerical_features)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.15, stratify=y, random_state=21) # X_poly_filtered

from imblearn.over_sampling import RandomOverSampler

# Before scaling
ros = RandomOverSampler(random_state=222)
X_train, y_train = ros.fit_resample(X_train, y_train)

print("Class distribution:", np.bincount(y_train))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Build the neural network with trial parameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3533316113774422),
    Dense(32, activation='relu'),
    Dropout(0.10179447781466523),
    Dense(16, activation='linear'),
    Dense(1)
])

nn_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop_best = EarlyStopping(monitor='val_loss')

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop_best],
    verbose=2
)

  # Predict probabilities
nn_proba = nn_model.predict(X_test_scaled).flatten()

# Determine best threshold based on ROC curve
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, nn_proba)
best_thresh_nn = thresholds_nn[(tpr_nn - fpr_nn).argmax()]

# Final prediction using best threshold
y_pred_final_nn = (nn_proba >= best_thresh_nn).astype(int)
auc_nn = roc_auc_score(y_test, nn_proba)

print(f"Best threshold for the neural network: {best_thresh_nn:.2f}")
print(f"Neural Net Accuracy: {accuracy_score(y_test, y_pred_final_nn):.4f}")
print(f"Neural Net AUC: {auc_nn:.4f}")
print("\nNeural Net Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final_nn))
print("\nNeural Net Classification Report:")
print(classification_report(y_test, y_pred_final_nn, target_names=["Same/Declined", "Improved"]))

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.grid(True)
plt.show()

# Reconstruct prediction labels using the best threshold
best_pred = (nn_proba >= best_thresh_nn).astype(int)

# Create a copy of the test portion of the original data
summary_df = data.iloc[y_test.index].copy()

# Add predicted and actual class labels
summary_df['Predicted'] = np.where(best_pred == 1, "Improved", "Same/Declined")
summary_df['Actual'] = np.where(y_test == 1, "Improved", "Same/Declined")

# Select only the relevant columns for display
ranking_comparison_df = summary_df[['University', 'ME Ranking 2023', 'ME Ranking 2024', 'Actual', 'Predicted']]

# export the comparison DataFrame
ranking_comparison_df[['University', 'ME Ranking 2023', 'ME Ranking 2024', 'Actual', 'Predicted']].to_csv("prediction_output_XGBoost.csv", index=False)
