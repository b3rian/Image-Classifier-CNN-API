import streamlit as st
import plotly.express as px
import pandas as pd

class UIComponents:
    @staticmethod
    def results_view(predictions: list):
        """Display predictions with interactive charts"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                "Top Prediction",
                value=predictions[0]['class_label'],
                delta=f"{predictions[0]['confidence']*100:.2f}%"
            )
        
        with col2:
            df = pd.DataFrame(predictions)
            fig = px.bar(
                df,
                x='confidence',
                y='class_label',
                orientation='h',
                color='confidence'
            )
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def feedback_system(predictions: list):
        """Enhanced feedback interface with correction input"""
        if 'predictions' not in st.session_state:
            return
        
        with st.expander("✏️ Provide Feedback", expanded=False):
            cols = st.columns([1, 1, 2])
            
            # Positive feedback
            with cols[0]:
                if st.button("👍 Correct", help="The prediction was accurate"):
                    FeedbackLogger.log_feedback(predictions, positive=True)
                    st.session_state.last_feedback = "positive"
                    st.toast("Thank you for confirming!", icon="✅")
            
            # Negative feedback
            with cols[1]:
                if st.button("👎 Incorrect", help="The prediction was wrong"):
                    st.session_state.show_correction = True
            
            # Correction input (only appears after negative feedback)
            if st.session_state.get('show_correction'):
                with cols[2]:
                    correction = st.text_input(
                        "What should the correct label be?",
                        placeholder="Enter the correct class...",
                        key="correction_input"
                    )
                    if correction:
                        FeedbackLogger.log_feedback(
                            predictions,
                            positive=False,
                            correction=correction
                        )
                        st.session_state.show_correction = False
                        st.session_state.last_feedback = "negative"
                        st.toast("Thanks for helping us improve!", icon="📝")
                        st.rerun()  # Refresh to show updated feedback state
