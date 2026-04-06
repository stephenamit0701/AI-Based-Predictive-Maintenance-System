# machine_fault_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# ========= CONFIG ==========
DATA_PATH = "machine_fault_dataset.csv"  # Your dataset path
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(RESULTS_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(RESULTS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========= LOAD DATA ==========
df = pd.read_csv(DATA_PATH)
print("\n=== Head of Dataset ===")
print(df.head())

print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Class Distribution ===")
print(df['fault_class'].value_counts())

# ========= EDA PLOTS ==========
# Histogram for each feature
for col in ['temperature', 'humidity', 'oil_level', 'gas_value']:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(os.path.join(RESULTS_DIR, f"{col}_distribution.png"))
    plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"))
plt.close()

# ========= DATA PREPROCESS ==========
X = df[['temperature', 'humidity', 'oil_level', 'gas_value']].values
y = df['fault_class'].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use in standalone prediction
joblib.dump(scaler, SCALER_PATH)

# Reshape for LSTM (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# One-hot encode labels
y_encoded = to_categorical(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# ========= MODEL ==========
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_encoded.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ========= TRAIN ==========
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

# ========= SAVE MODEL ==========
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")

# ========= PLOT TRAINING PERFORMANCE ==========
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "training_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
plt.close()

# ========= EVALUATE ==========
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

# ========= ROC & PR CURVES ==========
n_classes = y_encoded.shape[1]
lb = LabelBinarizer()
lb.fit(y_true)
y_true_bin = lb.transform(y_true)

# ROC Curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"))
plt.close()

# Precision-Recall Curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
    plt.plot(recall, precision, label=f"Class {i} (AP={ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curves.png"))
plt.close()

# ========= SAVE METRICS ==========
metrics = {
    "accuracy": accuracy,
    "classes": int(n_classes),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"\n📊 Metrics saved at: {METRICS_PATH}")
print(f"📁 All results saved in: {RESULTS_DIR}")
