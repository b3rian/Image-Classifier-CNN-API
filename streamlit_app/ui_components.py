"""
UI components and display functions for the Streamlit app.
"""

import pandas as pd
import streamlit as st
import time

def display_predictions(predictions, model_version, inference_time):
    """
    Display classification predictions in the Streamlit UI.
    
    Args:
        predictions: List of prediction dictionaries
        model_version: Version of the model used
        inference_time: Time taken for inference in seconds
    """
    st.subheader(f"Predictions: {model_version}")
    if not predictions:
        st.warning("No predictions above the confidence threshold.")
        return
    df = pd.DataFrame(predictions)
    df = df.set_index("label")

    for pred in predictions:
        st.markdown(f"**{pred['label']}**: {pred['confidence']}%")
        st.progress(pred['confidence'] / 100.0)

    st.caption(f"Inference time: {inference_time:.2f}s") 

def setup_sidebar():
    """
    Setup and render the sidebar UI components.
    
    Returns:
        tuple: (model_name, num_predictions, confidence_threshold, compare_models)
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Preferences & Model Selection")
        with st.expander("Advanced Options"):
            num_predictions = st.slider(
                "Number of predictions", 
                1, 10, 3,
                help="""Set how many predictions to display (1-10). 
                Higher values show more alternatives but may include less relevant results."""
            )
            confidence_threshold = st.slider(
                "Confidence threshold (%)", 
                0, 100, 0,
                help="""Minimum confidence percentage (0-100%) required to show a prediction. 
                Increase to filter out low-confidence results."""
            )
            compare_models = st.checkbox(
                "🔁 Compare Models", 
                help="Run both models on the image and compare their predictions."
            )

        model_name = st.selectbox(
            "Select 🧠 AI Model", 
            ["efficientnet", "resnet"], 
            disabled=compare_models,
            help="""Choose a deep learning architecture: 
            • **EfficientNet:** Lightweight and fast (good for mobile/edge devices)
            • **ResNet:** Powerful general-purpose model (best accuracy/speed balance).
            Disabled when 'Compare Models' is active - all models will run simultaneously."""
        )

        st.markdown("---")
        st.subheader("💬 Feedback")

        with st.form("feedback_form_sidebar"):
            history = st.session_state["history"]
            if history:
                selected = st.selectbox("Select image to review", [h["name"] for h in history],
                help="""Choose a previously classified image to provide feedback on. 
                The model's predictions for this image will be shown below for reference.
                Only images with existing classification results appear here.""")
                rating = st.select_slider("Rating (1-5)", options=[1, 2, 3, 4, 5], value=3,
                help="""Rate the model's accuracy for this image:
                1 = Completely wrong • 2 = Mostly incorrect • 3 = Partially correct
                4 = Mostly accurate • 5 = Perfect prediction """)
                selected_item = next((h for h in history if h["name"] == selected), None)
                if selected_item:
                    st.markdown("**Model Predictions:**")
                    for pred in selected_item["predictions"]:
                        st.markdown(f"- {pred['label']}: {pred['confidence']:.1f}%")
                correction = st.text_input("Suggested correction", placeholder="Correct label",
                help="""If the AI's prediction was wrong, please provide:
                • The accurate label for this image
                • Be specific (e.g., 'Golden Retriever' instead of just 'Dog')
                • Use singular nouns where applicable
                Your input helps train better models!""")
                comment = st.text_area("Additional comments", placeholder="Anything else?",
                help="""Share details to improve the model:
                • What features did the AI miss?
                • Was the mistake understandable?
                • Any edge cases we should know about?
    
 (Examples: 'The turtle was partially obscured' or 'Confused labrador with golden retriever')""")
            else:
                st.info("No images classified yet.")
                selected = rating = correction = comment = None

            if st.form_submit_button("Submit Feedback", type='primary') and selected:
                st.session_state["feedback"][selected] = {
                    "rating": rating,
                    "predictions": selected_item.get("predictions", []),
                    "correction": correction,
                    "comment": comment,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.toast("Feedback saved!", icon="✅")

    return model_name, num_predictions, confidence_threshold, compare_models