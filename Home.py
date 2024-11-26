import streamlit as st
import datetime

from utils.navigation import add_navigation

# Configure the page
st.set_page_config(
    page_title="Page Title - Horse Racing Analytics",
    page_icon="[appropriate icon]",
    layout="wide"
)


# Header
st.title("Horse Racing Analytics")
st.subheader("Transform your racing insights with data")

# Main features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Research Hub")
    st.write("""
        - Market analysis
        - Betting strategies
        - Academic papers
    """)
    
with col2:
    st.markdown("### 🧮 Analytics Center")
    st.write("""
        - Odds calculator
        - Risk assessment
        - Performance metrics
    """)
    
with col3:
    st.markdown("### 📅 Event Tracker")
    st.write("""
        - Major events
        - Historical trends
        - Prize analysis
    """)

# Interactive section
st.markdown("## Today's Highlights")
tab1, tab2, tab3 = st.tabs(["Key Metrics", "Quick Actions", "Updates"])

with tab1:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Races Analyzed", "1,234", "23")
    with metric_col2:
        st.metric("Success Rate", "67%", "2%")
    with metric_col3:
        st.metric("Active Models", "5", "1")
    with metric_col4:
        st.metric("Last Update", "Today", "5m ago")

with tab2:
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        st.selectbox("Jump to Analysis", [
            "Select analysis type...",
            "Race Prediction",
            "Odds Analysis",
            "Track Conditions",
            "Jockey Performance"
        ])
    
    with action_col2:
        st.selectbox("Quick Tools", [
            "Select tool...",
            "Odds Calculator",
            "Risk Assessor",
            "Portfolio Tracker",
            "Event Calendar"
        ])

with tab3:
    st.markdown("""
        #### Latest Updates
        - New Model Released: Enhanced prediction for sprint races
        - Data Update: Historical records updated through April 2024
        - New Feature: Advanced filtering in race analysis
    """)

# Tips
with st.expander("Getting Started"):
    st.write("""
        1. Check today's key metrics
        2. Review recent model predictions
        3. Analyze upcoming events
        4. Compare historical data
    """)

# Simple footer
st.markdown("---")
st.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")