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
