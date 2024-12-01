import streamlit as st

st.set_page_config(page_title="Available Data", page_icon="Book", layout="wide")

st.title("Available Data")

st.markdown("""
- Race length
- Track type
- Weather conditions
- Jockey
- Trainer
- Owner
- Placing position
- Time taken
- Odds t seconds before the start of the race

# Data sources
## https://www.proformracing.com/
- Race statistics
- Owner winnings
- Jockey analysis
- Breeding data
- 
""")