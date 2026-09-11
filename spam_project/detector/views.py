from django.shortcuts import render
import pickle
from pathlib import Path


# ==========================================
# Load Model
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


with open(BASE_DIR / "model.pkl", "rb") as file:
    model = pickle.load(file)


with open(BASE_DIR / "vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# ==========================================
# Home Page
# ==========================================

def home(request):

    result = None
    is_spam = False
    confidence = None

    if request.method == "POST":

        email = request.POST.get("email", "").strip()

        if email:

            # Convert text into TF-IDF features
            email_vectorized = vectorizer.transform([email])

            # Prediction
            prediction = model.predict(email_vectorized)

            # Probability
            probabilities = model.predict_proba(email_vectorized)

            confidence = round(
                max(probabilities[0]) * 100,
                2
            )


            if prediction[0] == 1:

                result = "Spam Email"
                is_spam = True

            else:

                result = "Not Spam Email"
                is_spam = False


    return render(
        request,
        "home.html",
        {
            "result": result,
            "is_spam": is_spam,
            "confidence": confidence
        }
    )