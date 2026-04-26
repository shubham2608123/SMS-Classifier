import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- Setup --------------------

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------- Load Model --------------------

model = pickle.load(open(r'C:\Users\Asus\PycharmProjects\SMS Classifier\model.pkl', 'rb'))
vectorizer = pickle.load(open(r'C:\Users\Asus\PycharmProjects\SMS Classifier\vectorizer.pkl', 'rb'))

# Get expected feature count from model
expected_features = getattr(model, 'n_features_in_', None)
if expected_features is None:
    # Fallback for older sklearn versions
    expected_features = model.feature_log_prob_.shape[1]

# -------------------- Text Processing --------------------

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

# -------------------- Keyword-based Spam Detection --------------------

def is_spam_by_keywords(text):
    """Fallback keyword-based spam detection for obvious spam patterns."""
    text_lower = text.lower()
    
    # Strong spam indicators (combinations)
    spam_patterns = [
        ("you have won", "won"),
        ("you've won", "won"),
        ("u have won", "won"),
        ("you won", "won"),
        ("claim your prize", "claim"),
        ("claim your reward", "claim"),
        ("claim now", "claim"),
        ("selected for a prize", "selected"),
        ("your account has been selected", "selected"),
        ("congratulations", "won"),
        ("congratulations", "selected"),
        ("congratulations", "prize"),
        ("congratulations", "winner"),
        ("congratulations", "awarded"),
        ("urgent", "prize"),
        ("urgent", "won"),
        ("urgent", "winner"),
        ("urgent", "cash"),
        ("urgent", "selected"),
        ("act now", "win"),
        ("act now", "prize"),
        ("act now", "offer"),
        ("limited time", "offer"),
        ("limited time", "win"),
        ("limited time", "prize"),
        ("winner", "selected"),
        ("winner", "chosen"),
        ("winner", "awarded"),
        ("free entry", "win"),
        ("free entry", "prize"),
        ("100% free", "free"),
        ("no obligation", "free"),
        ("risk free", "free"),
        ("click here", "win"),
        ("click here", "prize"),
        ("call now", "claim"),
        ("call now", "prize"),
        ("call now", "win"),
        ("text", "to"),
        ("cash prize", "prize"),
        ("cash reward", "reward"),
    ]
    
    # Check for pattern combinations
    for pattern in spam_patterns:
        if all(p in text_lower for p in pattern):
            return True, f"Matched keyword pattern: {' + '.join(pattern)}"
    
    # Single strong indicators
    strong_indicators = [
        "you've won a",
        "you have won a",
        "you won a",
        "call to claim",
        "claim your prize now",
        "winner of",
        "awarded a prize",
        "selected as a winner",
        "free cash prize",
        "urgent: you have won",
    ]
    
    for indicator in strong_indicators:
        if indicator in text_lower:
            return True, f"Matched strong indicator: '{indicator}'"
    
    return False, None

# -------------------- Page Config --------------------

st.set_page_config(page_title="Spam Classifier", layout="centered")

# -------------------- Custom CSS --------------------

st.markdown("""
<style>
body {
    background-color: #0B0F19;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}
textarea {
    background-color: #2A2F3A !important;
    color: white !important;
    border-radius: 10px !important;
}
.stButton>button {
    border: 2px solid red;
    color: red;
    background-color: transparent;
    border-radius: 8px;
    height: 40px;
    width: 120px;
    font-weight: bold;
}
.result-spam {
    color: red;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}
.result-ham {
    color: #00FFAA;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- UI --------------------

st.markdown('<div class="title">📩 Email/SMS Spam Classifier</div>', unsafe_allow_html=True)

input_sms = st.text_area("Enter the message")

# -------------------- Prediction --------------------

if st.button("Predict"):
    if input_sms.strip() != "":
        transformed = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed])
        
        # Handle feature mismatch between vectorizer and model
        if vector_input.shape[1] != expected_features:
            vector_input = vector_input[:, :expected_features]
        
        model_result = model.predict(vector_input)[0]
        model_proba = model.predict_proba(vector_input)[0]
        
        # Keyword-based fallback for obvious spam
        keyword_result, keyword_reason = is_spam_by_keywords(input_sms)
        
        # Final result: spam if either model or keywords say spam
        is_spam = (model_result == 1) or keyword_result
        
        if is_spam:
            st.markdown('<div class="result-spam">🚨 Spam</div>', unsafe_allow_html=True)
            if keyword_result and model_result != 1:
                st.caption("ℹ️ Detected by keyword rules")
            elif model_result == 1:
                st.caption("ℹ️ Detected by ML model")
        else:
            st.markdown('<div class="result-ham">✅ Not Spam</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message first!")

