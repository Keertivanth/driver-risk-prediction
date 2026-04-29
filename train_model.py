import pandas as pd
from xgboost import XGBClassifier
import pickle

from step2_preprocessing import preprocess_data
from step3_features import create_features

df = pd.read_csv("train_motion_data.csv")

print("📊 Raw Data:")
print(df.head())

# preprocess
df = preprocess_data(df)

# target
if 'Class' not in df.columns:
    raise ValueError("❌ Class column missing")

y = df['Class']

# features
X = create_features(df)

print("\n🧪 Features used:")
print(X.columns)

# check classes
if len(set(y)) < 2:
    raise ValueError("❌ Need BOTH NORMAL and AGGRESSIVE classes")

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully")