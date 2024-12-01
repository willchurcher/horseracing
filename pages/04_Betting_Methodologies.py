import streamlit as st

st.title("Betting Methodologies")

st.markdown("""

# Modelling
Three main methods come to mind:
1. Time series modelling with Kalman filters
2. Generalised linear models
3. Monte Carlo simulation

For Kalman filters, we can estimate the finishing time of a horse based on on
- Handicap
- Jockey strength rating from the filter state
- Trainer strength rating from the filter state
- Horse fundamental speed from the filter state
- Horse endurance from the filter state

These can then be used to estimate the finishing time of a horse combined with the track type, distance, and weather conditions.

Leveraging data using GPS data on horse position (TODO: source), horse interaction dynamics can be estimated. Horses will interact, slowing down.
These effects can be understood.

This provides a model for the strength of horses.

# Modelling the market

The market's opinion on the outcome of the race can also be understood, by modelling the market win probabilities as a function of our horse
strength ratings. The outcome of a bet is based on the interaction between our opinion and the market's opinion, and we need to find horses which are
undervalued, which we are confident we will obtain a good payout from. We can therefore model the market's opinion, and the difference between our
opinion and the market's opinion.




""")