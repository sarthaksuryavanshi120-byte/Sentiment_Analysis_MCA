# sentiment.py
from transformers import pipeline

# Load sentiment analysis model
_senti_pipe = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Sarcasm phrases
_sarcasm_triggers = [
    "wow", "great", "good work", "amazing", "excellent",
    "brilliant", "fantastic", "lovely", "thanks a lot"
]
_negative_hints = [
    "delay", "twice", "before needed", "as usual",
    "late again", "more paperwork", "extra burden",
    "waste of time", "not helpful", "slow process"
]

# Mixed sentiment detection keywords
_positive_words = ["good", "great", "excellent", "amazing", "fantastic", "helpful"]
_negative_words = ["bad", "delay", "problem", "confusing", "burden", "waste", "slow"]


def _detect_rule_based_sarcasm(text: str) -> bool:
    """Detect sarcasm using simple rules."""
    lower_text = text.lower()
    return any(t in lower_text for t in _sarcasm_triggers) and any(n in lower_text for n in _negative_hints)


def _detect_mixed_sentiment(text: str) -> bool:
    """Detect mixed sentiment based on positive & negative words."""
    lower_text = text.lower()
    has_pos = any(p in lower_text for p in _positive_words)
    has_neg = any(n in lower_text for n in _negative_words)
    return has_pos and has_neg


def analyze_sentiment(text: str, neutral_threshold: float = 0.60):
    """Return sentiment label and score for a given text."""
    if not text or not text.strip():
        return "Neutral", 0.0

    # 1️⃣ Sarcasm
    if _detect_rule_based_sarcasm(text):
        return "Negative", 0.95

    # 2️⃣ Transformer-based sentiment
    try:
        res = _senti_pipe(text[:512])
        if not res or not isinstance(res, list):
            return "Neutral", 0.0

        top = res[0]
        label = top.get("label", "")
        score = float(top.get("score", 0.0))

        # 3️⃣ Mixed detection
        if _detect_mixed_sentiment(text):
            return "Mixed", score

        # 4️⃣ Neutral if low confidence
        if score < neutral_threshold:
            return "Neutral", score

        # 5️⃣ Positive / Negative
        if label.upper().startswith("POS"):
            return "Positive", score
        elif label.upper().startswith("NEG"):
            return "Negative", score
        else:
            return "Neutral", score

    except Exception:
        return "Neutral", 0.0
