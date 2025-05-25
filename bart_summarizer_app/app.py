import gradio as gr
from transformers import pipeline

# Load summarization pipeline using facebook/bart-base
summarizer = pipeline("summarization", model="facebook/bart-base")

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

description = """
Enter a news article, legal document, or long paragraph and receive a concise summary.
This app uses the `facebook/bart-base` model from Hugging Face Transformers.
"""

# Gradio UI
demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, placeholder="Paste your text here..."),
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="BART-based Text Summarizer",
    description=description,
)

demo.launch()
