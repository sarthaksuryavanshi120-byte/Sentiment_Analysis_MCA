from transformers import pipeline

# Download the smaller model
summarizer = pipeline("summarization", model="t5-small")
print("Model downloaded successfully!")
