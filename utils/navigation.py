# utils/navigation.py
import streamlit as st

def add_navigation():
    with st.sidebar:
        st.title("Navigation")
        st.markdown("""
            - [🏠 Home](Home)
            - [📚 Literature Review](01_literature_review)
            - [🎯 Betting Calculator](Betting_Calculator)
            - [📅 Major Events](Major_Events)
            - [📊 Data Analysis](Data_Analysis)
            - [🤖 Predictive Models](Predictive_Models)
            - [👤 Portfolio](Portfolio)
        """)