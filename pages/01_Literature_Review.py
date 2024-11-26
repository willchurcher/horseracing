import streamlit as st
import pandas as pd
from utils.navigation import add_navigation

st.set_page_config(page_title="Literature Review - Horse Racing Analytics", page_icon="📚", layout="wide")
# Define your citations as a list of dictionaries
citations = [
    {
        "title": "The Timing of Parimutuel Bets",
        "authors": "Macro Ottaviani & Peter Norman Sørensen",
        "year": 2006,
        "url": "https://didattica.unibocconi.it/mypage/upload/48832_20130621_094938_TOBAFLB.PDF"
    }
]

st.title("Literature Review")

# Display citations as clickable links
for citation in citations:
    st.markdown(
        f"[{citation['title']}]({citation['url']}) - {citation['authors']} ({citation['year']})"
    )


st.header("The Timing of Parimutuel Bets")
st.markdown("""
The authors explain three observable phenomena:
1. A sizable fraction of bets is placed early.
2. Late bets are more informative than early bets.
3. Proportionally too many bets are placed on longshots.

They explain this behaviour by examining the market dynamics of the behaviour of two types of bettors:
1. Large bettors who act on common information are incentivised to bet early.
2. Small bettors who act on private information are incentivised to bet late.

This then also explains the third phenomenon, that too many bets are placed on longshots.

""")
