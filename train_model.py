import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# === LOAD DATA ===
df = pd.read_csv("features.csv")
print("✅ Loaded dataset:", df.shape)

# === SPLIT FEATURES AND LABELS ===
X = df.drop("label", axis=1)
y = df["label"]

# === ENCODE LABELS ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === NORMALIZE FEATURES ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# === CHOOSE YOUR MODEL ===
# You can switch between SVM, RandomForest, or KNN
# Example 1: Support Vector Machine
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

# Example 2: Random Forest
# model = RandomForestClassifier(n_estimators=200, random_state=42)

# Example 3: K-Nearest Neighbors
# model = KNeighborsClassifier(n_neighbors=5)

# === TRAIN ===
print("\n🚀 Training the model...")
model.fit(X_train_scaled, y_train)

joblib.dump(model, "genre_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("💾 Saved model and label encoder.")


# === PREDICT ===
y_pred = model.predict(X_test_scaled)

# === EVALUATE ===
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc * 100:.2f}%\n")

# === CLASSIFICATION REPORT ===
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Accuracy: {acc*100:.2f}%)")
plt.show()