import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\salary_prediction_app\data\adult.csv")

# Replace '?' with NaN and drop rows with missing values
df.replace(' ?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Define target
target_col = "income"
X = df.drop(columns=[target_col])
y = df[target_col]

# Separate categorical and numeric columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(exclude='object').columns.tolist()

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model directory
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump({
    "encoders": label_encoders,
    "scaler": scaler,
    "target_encoder": target_encoder
}, "model/encoders.pkl")

print("✅ Model and preprocessing tools saved in 'model/' folder.")
