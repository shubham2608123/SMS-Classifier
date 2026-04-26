# 📩 Email/SMS Spam Classifier

An AI-powered web application that classifies messages as **Spam 🚨** or **Not Spam ✅** using Machine Learning and Natural Language Processing.

---

## 🚀 Features

* 🔍 Detects spam messages in real-time
* 🧠 Uses trained ML model (Naive Bayes)
* ✨ Clean and modern UI with Streamlit
* ⚡ Fast and lightweight prediction
* 📝 Text preprocessing (tokenization, stopword removal, stemming)

---

## 🧠 Tech Stack

* Python 🐍
* Streamlit 🌐
* Scikit-learn 🤖
* NLTK 📚
* Pickle (model serialization)

---

## 📁 Project Structure

```
project/
│
├── app.py              # Streamlit frontend
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Dependencies
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Download NLTK data (run once)

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4️⃣ Run the app

```
streamlit run app.py
```

---

## 🧪 Example Inputs

### 🔴 Spam

* "Congratulations! You have won a prize. Claim now!"
* "Free entry in a competition. Text WIN now!"

### 🔵 Not Spam (Ham)

* "Hey, are we meeting today?"
* "Call me when you reach home"

---

## 🧠 How it Works

1. Input message is cleaned and processed
2. Text is transformed using TF-IDF vectorizer
3. Trained Naive Bayes model predicts the result
4. Output is displayed as Spam or Not Spam

---

## 📊 Model Details

* Algorithm: Multinomial Naive Bayes
* Vectorization: TF-IDF
* Text Processing:

  * Lowercasing
  * Tokenization
  * Stopword removal
  * Stemming

---

## 🎯 Future Improvements

* 📈 Show prediction probability
* 🌐 Deploy online (Streamlit Cloud)
* 🧠 Improve accuracy with advanced models
* 📊 Add analytics dashboard

---

## 🙌 Author

**Soham Dalvi**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
