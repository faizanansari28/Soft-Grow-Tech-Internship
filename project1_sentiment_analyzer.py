# ============================================================
#   PROJECT 1 – Text Sentiment Analyzer
#   SoftGrowTech Internship | Task 2
#   Uses: TextBlob + NLTK (VADER)
# ============================================================

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required NLTK data (runs once)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)


# ─────────────────────────────────────────────
#  Helper: Classify using VADER (NLTK)
# ─────────────────────────────────────────────
def analyze_with_vader(text: str) -> dict:
    """
    Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).
    Best for short social-media style text.
    Returns compound score + label.
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)

    compound = scores['compound']
    if compound >= 0.05:
        label = "POSITIVE 😊"
    elif compound <= -0.05:
        label = "NEGATIVE 😞"
    else:
        label = "NEUTRAL 😐"

    return {
        "label":    label,
        "compound": round(compound, 4),
        "positive": round(scores['pos'], 4),
        "negative": round(scores['neg'], 4),
        "neutral":  round(scores['neu'], 4),
    }


# ─────────────────────────────────────────────
#  Helper: Classify using TextBlob
# ─────────────────────────────────────────────
def analyze_with_textblob(text: str) -> dict:
    """
    Uses TextBlob pattern-based analysis.
    Returns polarity (-1 to 1) and subjectivity (0 to 1).
    """
    blob = TextBlob(text)
    polarity    = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.05:
        label = "POSITIVE 😊"
    elif polarity < -0.05:
        label = "NEGATIVE 😞"
    else:
        label = "NEUTRAL 😐"

    return {
        "label":        label,
        "polarity":     round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
    }


# ─────────────────────────────────────────────
#  Display results in a formatted box
# ─────────────────────────────────────────────
def display_results(text: str, vader: dict, tb: dict) -> None:
    print("\n" + "=" * 55)
    print("  SENTIMENT ANALYSIS RESULTS")
    print("=" * 55)
    print(f"  Input Text : {text[:60]}{'...' if len(text) > 60 else ''}")
    print("-" * 55)
    print("  ── VADER (NLTK) ──────────────────────────────")
    print(f"  Sentiment  : {vader['label']}")
    print(f"  Compound   : {vader['compound']:+.4f}  "
          f"(Pos={vader['positive']} | Neg={vader['negative']} | Neu={vader['neutral']})")
    print("-" * 55)
    print("  ── TextBlob ──────────────────────────────────")
    print(f"  Sentiment  : {tb['label']}")
    print(f"  Polarity   : {tb['polarity']:+.4f}  "
          f"(Subjectivity={tb['subjectivity']:.4f})")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  Demo with sample sentences
# ─────────────────────────────────────────────
def run_demo():
    samples = [
        "I absolutely love this product! It exceeded all my expectations.",
        "This is the worst experience I have ever had. Totally disappointed.",
        "The package arrived on Tuesday.",
        "The movie was okay, not great but not terrible either.",
        "Amazing service! Will definitely recommend to all my friends.",
        "I hate waiting in long queues, it is so frustrating!",
    ]

    print("\n" + "★" * 55)
    print("   TEXT SENTIMENT ANALYZER  –  SoftGrowTech Task 2")
    print("★" * 55)
    print("  Running demo on sample sentences …\n")

    for sentence in samples:
        vader_result = analyze_with_vader(sentence)
        tb_result    = analyze_with_textblob(sentence)
        display_results(sentence, vader_result, tb_result)


# ─────────────────────────────────────────────
#  Interactive mode
# ─────────────────────────────────────────────
def interactive_mode():
    print("\n" + "─" * 55)
    print("  INTERACTIVE MODE  (type 'quit' to exit)")
    print("─" * 55)

    while True:
        text = input("\n  Enter text to analyze: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("  Goodbye! 👋")
            break
        if not text:
            print("  ⚠  Please enter some text.")
            continue

        vader_result = analyze_with_vader(text)
        tb_result    = analyze_with_textblob(text)
        display_results(text, vader_result, tb_result)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_demo()          # Show built-in demo first
    interactive_mode()  # Then let user type their own text
