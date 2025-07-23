
# Install Streamlit first:
#pip install streamlit transformers torch


import streamlit as st
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import plotly.graph_objects as go

# Load model & tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Prediction function
def predict_dark_triad(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze()
    traits = ['Machiavellianism', 'Narcissism', 'Psychopathy']
    return {trait: float(prob) for trait, prob in zip(traits, probs)}


def draw_spider_chart(scores):
    categories = list(scores.keys())
    values = list(scores.values())
    values += values[:1]  # Close the radar chart loop
    categories += categories[:1]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=values, theta=categories, fill='toself', name="Personality Profile", line_color='darkred')
        ],
        layout=go.Layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False
        )
    )
    st.plotly_chart(fig)


# Web UI
st.title("🧠 XP Lab -- Dark Triad Text Analyzer")
text_input = st.text_area("Enter text to analyze:", "I always get what I want, no matter what it takes.")

if st.button("Analyze"):
    scores = predict_dark_triad(text_input)
    st.subheader("Personality Trait Probabilities:")
    for trait, score in scores.items():
        st.write(f"**{trait}**: {score:.2%}")
        st.progress(score)
    st.subheader("🕷️ Spider Chart of Traits")
    draw_spider_chart(scores)

