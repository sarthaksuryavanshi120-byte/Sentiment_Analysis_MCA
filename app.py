# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import bleach
from collections import Counter
import io
import base64
import re
import string
from sentiment import analyze_sentiment  # your existing sentiment.py

# Download stopwords once
nltk.download('stopwords')

# Initialize summarizer
summarizer = pipeline("summarization", model="t5-small")

# Allowed HTML tags in comments
ALLOWED_TAGS = ["b", "i", "u", "em", "strong", "br"]

stop_words = set(stopwords.words("english"))

# -------------------
# Helper Functions
# -------------------

def summarize_comment(comment, max_words=50):
    text = (comment or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    try:
        summary = summarizer(text, max_length=15, min_length=5, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return " ".join(words[:10]) + ("..." if len(words) > 10 else "")

def summarize_of_summaries(summaries):
    if not summaries:
        return ""
    text = " ".join(summaries)
    try:
        summary = summarizer(text[:500], max_length=40, min_length=12, do_sample=False)
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
            highlighted.append(f"**{w}**")
        else:
            highlighted.append(w)
    return " ".join(highlighted)

def generate_wordcloud(texts, max_words=100):
    combined = " ".join([t for t in texts if t])
    combined = re.sub(r'\s+', ' ', combined)
    stopwords_set = set(STOPWORDS)
    wc = WordCloud(width=800, height=400, background_color="white",
                   stopwords=stopwords_set, max_words=max_words)
    wc.generate(combined)
    return wc.to_array()

# -------------------
# Streamlit App
# -------------------

st.title("Sentiment Analysis MCA")

# Input options
input_option = st.radio("Select Input Method:", ["Paste Text", "Upload CSV"])

comments_list = []

if input_option == "Paste Text":
    text_input = st.text_area("Enter comments (one per line):")
    if text_input:
        raw_comments = [line.strip() for line in text_input.splitlines() if line.strip()]
        comments_list = [bleach.clean(c, tags=ALLOWED_TAGS, strip=True) for c in raw_comments]

elif input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with column 'Comment_Text'", type="csv")
    if uploaded_file:
        try:
            df_csv = pd.read_csv(uploaded_file)
            if "Comment_Text" not in df_csv.columns:
                st.error("CSV must contain a column named 'Comment_Text'")
            else:
                comments_list = [bleach.clean(str(c), tags=ALLOWED_TAGS, strip=True)
                                 for c in df_csv["Comment_Text"].fillna("")]
        except Exception:
            st.error("Could not read CSV. Please upload a valid CSV.")

# Perform analysis if comments exist
if comments_list:
    st.subheader("Analysis Results")
    
    # Sentiment Analysis
    sentiments_list = []
    summaries = []
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}

    for comment in comments_list:
        label, score = analyze_sentiment(comment)
        sentiments_list.append(label)
        counts[label] = counts.get(label, 0) + 1
        summaries.append(summarize_comment(comment))

    # Display sentiment counts
    st.write(f"Positive: {counts['Positive']}, Negative: {counts['Negative']}, Neutral: {counts['Neutral']}, Mixed: {counts['Mixed']}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.keys(), counts.values(), color=["#4CAF50","#F44336","#FFC107","#2196F3"])
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Comments")
    st.pyplot(fig)

    # WordCloud
    st.subheader("Word Cloud")
    wc_image = generate_wordcloud(comments_list)
    st.image(wc_image)

    # Summary of summaries
    st.subheader("Summary of All Comments")
    summary_of_summaries = summarize_of_summaries(summaries)
    highlighted_summary = highlight_text(summary_of_summaries, " ".join(summaries))
    st.markdown(highlighted_summary)

    # Download CSV
    st.subheader("Download Analyzed CSV")
    df_out = pd.DataFrame({
        "Comment_Text": comments_list,
        "Sentiment": sentiments_list,
        "Summary": summaries
    })
    csv_data = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_data, file_name="analyzed_comments.csv", mime="text/csv")
