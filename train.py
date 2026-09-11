import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("Datasets/emails.csv")

print("Dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ==========================================
# 2. Separate Input and Output
# ==========================================

x = df["text"]
y = df["spam"]


# ==========================================
# 3. Split Dataset
# ==========================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. Convert Text into Numerical Features
# ==========================================

vectorizer = TfidfVectorizer()

x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


# ==========================================
# 5. Create and Train Model
# ==========================================

model = LogisticRegression()

model.fit(x_train_vectorized, y_train)


# ==========================================
# 6. Test Model
# ==========================================

prediction = model.predict(x_test_vectorized)


# ==========================================
# 7. Evaluate Model
# ==========================================

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, prediction))


# ==========================================
# 8. Save Model
# ==========================================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


print("\nModel saved successfully!")
print("Created: model.pkl")
print("Created: vectorizer.pkl")