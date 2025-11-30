# application.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊", # You can use an emoji here
    layout="wide", # Use more screen width
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
# This adds a nice background and styles our containers
st.markdown("""
<style>
.main {
    background-color: #F0F2F6;
}
.stTextArea {
    background-color: #FFFFFF;
    border-radius: 10px;
    padding: 10px;
}
.css-1d391kg {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)


# --- Configuration ---
# The path to your FULL fine-tuned model
MODEL_PATH = "bert-base-uncased-sentiment-model"

# The emotion labels from your config file
ID_TO_LABEL = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# --- Caching the Model Loading ---
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the full fine-tuned model and tokenizer."""
    with st.spinner("Loading model and tokenizer... This may take a minute."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return model, tokenizer

# --- Prediction Function ---
def predict_emotion(text, model, tokenizer):
    """Predicts the emotion of a given text and returns all probabilities."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1).squeeze().numpy()
    
    # Get probabilities for each emotion
    emotion_probs = {ID_TO_LABEL[i]: prob for i, prob in enumerate(probabilities)}
    
    # Sort by probability
    sorted_emotions = sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_emotions

# --- Main App Logic ---
def main():
    model, tokenizer = load_model_and_tokenizer()

    st.title("😊 Emotion Analysis App")
    st.markdown("Enter a sentence below to analyze its underlying emotion using a fine-tuned BERT model.")

    # --- User Input Section ---
    with st.container():
        user_input = st.text_area(
            "Enter your text here:",
            placeholder="e.g., I am so happy and excited to be here!",
            height=150
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear Text", use_container_width=True)

        if clear_button:
            st.rerun()
        
        if predict_button and user_input:
            sorted_emotions = predict_emotion(user_input, model, tokenizer)
            
            top_emotion, top_confidence = sorted_emotions[0]
            
            # --- Display Results ---
            st.markdown("---")
            st.subheader("Analysis Result")
            
            # Display top prediction in a nice metric card
            col_result, col_chart = st.columns([1, 2])
            with col_result:
                st.metric(
                    label="**Predicted Emotion**",
                    value=f"**{top_emotion.title()}**",
                    delta=f"Confidence: {top_confidence:.2%}"
                )
            
            # Create and display a bar chart for all emotions
            with col_chart:
                st.markdown("**Confidence Breakdown:**")
                emotions_df = pd.DataFrame(sorted_emotions, columns=["Emotion", "Confidence"])
                fig = px.bar(
                    emotions_df, 
                    x="Confidence", 
                    y="Emotion", 
                    orientation='h',
                    color="Confidence",
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                # --- THIS IS THE CORRECTED LINE ---
                fig.update_layout(height=300, yaxis={'categoryorder':'category ascending'})
                st.plotly_chart(fig, use_container_width=True)


# --- Sidebar Information ---
def sidebar():
    st.sidebar.title("ℹ️ App Information")
    st.sidebar.info(
        """
        This application uses a **BERT-base-uncased** model that has been fine-tuned on a dataset for multi-class emotion classification.
        
        **Model:** `bert-base-uncased-sentiment-model`
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Detected Emotions:**")
    for label in ID_TO_LABEL.values():
        st.sidebar.markdown(f"- {label.title()}")

# --- Run the App ---
if __name__ == "__main__":
    # We need pandas for the chart
    import pandas as pd
    sidebar()
    main()
