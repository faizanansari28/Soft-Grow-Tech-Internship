# 📊 Project 1 – Text Sentiment Analyzer

> **SoftGrowTech Internship | Task 2**

A Python-based sentiment analysis tool that evaluates the emotional tone of any text using **two independent NLP engines** — NLTK's VADER and TextBlob — and compares their results side by side.

---

## ✨ Features

- Dual-engine analysis: **VADER** (NLTK) + **TextBlob** running in parallel
- Classifies text as **POSITIVE 😊**, **NEGATIVE 😞**, or **NEUTRAL 😐**
- VADER returns compound, positive, negative, and neutral scores
- TextBlob returns polarity (−1 to +1) and subjectivity (0 to 1)
- Built-in **demo mode** with 6 sample sentences
- **Interactive mode** — type any text and get instant results
- Clean, formatted console output

---

## 🛠️ Technologies Used

| Library     | Purpose                              |
|-------------|--------------------------------------|
| `nltk`      | VADER sentiment scoring              |
| `textblob`  | Pattern-based polarity & subjectivity |

---

## ⚙️ Installation

```bash
pip install nltk textblob
```

NLTK data is downloaded automatically on first run (`vader_lexicon`, `punkt`).

---

## ▶️ How to Run

```bash
python project1_sentiment_analyzer.py
```

The script first runs a **demo** on 6 preset sentences, then enters **interactive mode** where you can type your own text.

---

## 📋 How It Works

```
Input Text
    │
    ├──▶ VADER (NLTK)
    │       • Compound score (−1 to +1)
    │       • Positive / Negative / Neutral breakdown
    │       • Best for short, social-media-style text
    │
    └──▶ TextBlob
            • Polarity score (−1 to +1)
            • Subjectivity score (0 to 1)
            • Best for formal / longer text
```

### Scoring thresholds

| Score        | Label    |
|--------------|----------|
| ≥ +0.05      | POSITIVE |
| ≤ −0.05      | NEGATIVE |
| Between both | NEUTRAL  |

---

## 🖥️ Sample Output

```
=======================================================
  SENTIMENT ANALYSIS RESULTS
=======================================================
  Input Text : I absolutely love this product! It exceeded all my...
-------------------------------------------------------
  ── VADER (NLTK) ──────────────────────────────
  Sentiment  : POSITIVE 😊
  Compound   : +0.6369  (Pos=0.38 | Neg=0.0 | Neu=0.62)
-------------------------------------------------------
  ── TextBlob ──────────────────────────────────
  Sentiment  : POSITIVE 😊
  Polarity   : +0.5000  (Subjectivity=0.6000)
=======================================================
```

---

## 📁 Project Structure

```
project1_sentiment_analyzer.py
├── analyze_with_vader()      # NLTK VADER scoring
├── analyze_with_textblob()   # TextBlob scoring
├── display_results()         # Formatted console output
├── run_demo()                # 6 built-in sample sentences
└── interactive_mode()        # User input loop
```

---

## 👤 Author

**Faizan Ansari** — SoftGrowTech Internship
