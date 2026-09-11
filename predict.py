import pickle


# ==========================================
# 1. Load Trained Model
# ==========================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================
# 2. Get Email from User
# ==========================================

email = input("Enter an email/message: ")


# ==========================================
# 3. Convert Email into Numerical Features
# ==========================================

email_vectorized = vectorizer.transform([email])


# ==========================================
# 4. Make Prediction
# ==========================================

result = model.predict(email_vectorized)


# ==========================================
# 5. Display Result
# ==========================================

if result[0] == 1:
    print("\nResult: Spam Email")
else:
    print("\nResult: Not Spam Email")