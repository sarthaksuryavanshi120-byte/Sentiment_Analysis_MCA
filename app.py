from flask import Flask, request, render_template, send_file
from sentiment import analyze_sentiment
import pandas as pd
import matplotlib.pyplot as plt
import io, base64, os, re, string
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import bleach
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Uncomment first run
# nltk.download("stopwords")

app = Flask(__name__)

# Initialize summarizer once
summarizer = pipeline("summarization", model="t5-small")

# Allowed HTML tags in comments
ALLOWED_TAGS = ["b", "i", "u", "em", "strong", "br"]

stop_words = set(stopwords.words("english"))


def summarize_comment(comment, max_words=50):
    text = (comment or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    try:
        summary = summarizer(
            text,
            max_length=15,
            min_length=5,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception:
        return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def summarize_of_summaries(summaries):
    if not summaries:
        return ""
    text = " ".join(summaries)
    try:
        summary = summarizer(
            text[:500],
            max_length=40,
            min_length=12,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception:
        return " ".join(text.split()[:30]) + "..."


def highlight_text(text, all_summaries, min_freq=2):
    words = [
        w.lower().strip(string.punctuation)
        for w in all_summaries.split()
        if w.lower() not in stop_words and len(w) > 2
    ]
    freq = Counter(words)
    key_terms = {word for word, count in freq.items() if count >= min_freq}

    highlighted = []
    for w in text.split():
        word_clean = w.lower().strip(string.punctuation)
        if word_clean in key_terms:
            highlighted.append(f"<mark>{w}</mark>")
        else:
            highlighted.append(w)
    return " ".join(highlighted)


def fig_to_base64(plt_figure):
    buf = io.BytesIO()
    plt_figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt_figure.clf()
    plt.close()
    return img_b64


def generate_wordcloud(texts, max_words=100):
    combined = " ".join([t for t in texts if t])
    combined = re.sub(r'\s+', ' ', combined)
    stopwords_set = set(STOPWORDS)
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=stopwords_set, max_words=max_words)
    wc.generate(combined)
    img_buf = io.BytesIO()
    wc.to_image().save(img_buf, format="PNG")
    img_buf.seek(0)
    return base64.b64encode(img_buf.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    chart_img = None
    wordcloud_img = None
    summaries = []
    comments_list = []
    sentiments_list = []
    highlighted_summary = ""

    if request.method == "POST":
        if "text" in request.form and request.form["text"].strip():
            text = request.form["text"]
            raw_comments = [line.strip() for line in text.splitlines() if line.strip()]
            comments_list = [bleach.clean(c, tags=ALLOWED_TAGS, strip=True) for c in raw_comments]
        else:
            comments_list = []

        counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}

        for comment in comments_list:
            label, score = analyze_sentiment(comment)
            sentiments_list.append(label)
            counts[label] = counts.get(label, 0) + 1
            summaries.append(summarize_comment(comment))

        if comments_list:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.keys(), counts.values(), color=["#4CAF50","#F44336","#FFC107","#2196F3"])
            ax.set_title("Sentiment Distribution")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Number of Comments")
            chart_img = fig_to_base64(fig)

            wordcloud_img = generate_wordcloud(comments_list)

        result = f"Positive: {counts['Positive']}, Negative: {counts['Negative']}, Neutral: {counts['Neutral']}, Mixed: {counts['Mixed']}"

        # People say's...
        summary_of_summaries = summarize_of_summaries(summaries)
        highlighted_summary = highlight_text(summary_of_summaries, " ".join(summaries))

    return render_template(
        "index.html",
        result=result,
        chart_img=chart_img,
        wordcloud_img=wordcloud_img,
        comments=comments_list or [],
        summaries=summaries or [],
        sentiments=sentiments_list or [],
        summary_of_summaries=highlighted_summary or ""
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return "No file uploaded!"

    try:
        df = pd.read_csv(file)
    except Exception:
        return "Error: Could not read CSV. Please upload a valid CSV file."

    if "Comment_Text" not in df.columns:
        return "Error: CSV must have a column named 'Comment_Text'."

    counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    sentiments_list = []
    summaries = []
    comments = []

    for comment in df["Comment_Text"].astype(str).fillna(""):
        safe_comment = bleach.clean(comment, tags=ALLOWED_TAGS, strip=True)
        comments.append(safe_comment)
        label, score = analyze_sentiment(safe_comment)
        sentiments_list.append(label)
        counts[label] = counts.get(label, 0) + 1
        summaries.append(summarize_comment(safe_comment))

    df["Sentiment"] = sentiments_list
    df["Summary"] = summaries

    out_path = "analyzed_comments.csv"
    df.to_csv(out_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.keys(), counts.values(), color=["#4CAF50","#F44336","#FFC107","#2196F3"])
    ax.set_title("Sentiment Distribution (CSV Upload)")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Comments")
    chart_img = fig_to_base64(fig)

    wordcloud_img = generate_wordcloud(comments)

    result = f"Positive: {counts['Positive']}, Negative: {counts['Negative']}, Neutral: {counts['Neutral']}, Mixed: {counts['Mixed']}"

    summary_of_summaries = summarize_of_summaries(summaries)
    highlighted_summary = highlight_text(summary_of_summaries, " ".join(summaries))

    return render_template(
        "index.html",
        result=result,
        chart_img=chart_img,
        wordcloud_img=wordcloud_img,
        comments=comments or [],
        summaries=summaries or [],
        sentiments=sentiments_list or [],
        summary_of_summaries=highlighted_summary or ""
    )


@app.route("/download", methods=["GET"])
def download():
    path = "analyzed_comments.csv"
    if not os.path.exists(path):
        return "No analyzed CSV found. Upload and analyze first."
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
