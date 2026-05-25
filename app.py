from flask import Flask, request, jsonify
import joblib
import urllib.parse
import html
import re

app = Flask(__name__)

# =========================================================
# LOAD MODEL & VECTORIZER
# =========================================================

model = joblib.load('random_forest_model_v6.pkl')

tfidf = joblib.load('tfidf_vectorizer_v6.pkl')


# =========================================================
# PREPROCESSING
# WAJIB SAMA DENGAN TRAINING
# =========================================================

def preprocess_text(text):

    text = str(text)

    text = text.lower()

    text = urllib.parse.unquote(text)

    text = html.unescape(text)

    text = text.replace('\t', ' [tab] ')

    text = text.encode(
        "ascii",
        "ignore"
    ).decode()

    text = re.sub(
        r'([<>=\'\"/().%;#\[\]])',
        r' \1 ',
        text
    )

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================================================
# HOME ROUTE
# =========================================================

@app.route('/')
def home():

    return "ML WAF SERVER ACTIVE"


# =========================================================
# PREDICT ROUTE
# =========================================================

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # =========================================
        # AMBIL JSON
        # =========================================

        data = request.get_json()

        payload = data.get(
            "payload",
            ""
        )

        print("\n============================")
        print("PAYLOAD ASLI:")
        print(payload)

        # =========================================
        # PREPROCESSING
        # =========================================

        clean_payload = preprocess_text(payload)

        print("\nSETELAH PREPROCESS:")
        print(clean_payload)

        # =========================================
        # TFIDF
        # =========================================

        vector = tfidf.transform(
            [clean_payload]
        )

        # =========================================
        # PREDIKSI
        # =========================================

        prediction = model.predict(vector)[0]

        confidence = (
            model.predict_proba(vector).max()
            * 100
        )

        print("\nPREDIKSI:")
        print(prediction)

        print("\nCONFIDENCE:")
        print(f"{confidence:.2f}%")

        # =========================================
        # DECISION
        # =========================================

        if prediction == "normal":

            status = "ALLOW"

        else:

            status = "BLOCK"

        print("\nSTATUS:")
        print(status)

        print("============================\n")

        # =========================================
        # RESPONSE
        # =========================================

        return jsonify({

            "status": status,

            "prediction": prediction,

            "confidence": round(
                confidence,
                2
            )
        })

    except Exception as e:

        print("ERROR :", str(e))

        return jsonify({

            "status": "ERROR",

            "message": str(e)

        })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
