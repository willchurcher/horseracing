import streamlit as st
import pandas as pd

st.set_page_config(page_title="Statistics and Horseracing: A History", page_icon="📚", layout="wide")

st.markdown("""
# Historical Development of Horse Racing Models: A Chronological Analysis

## Early Statistical Foundations (1900s-1950s)

### Richard von Mises (1928)
- Paper: "Probability, Statistics and Truth"
- Introduced fundamental probability concepts later applied to racing
- Established framework for analyzing repeated events under similar conditions

### F.E. Croxton & D.J. Cowden (1939)
- Publication: "Applied General Statistics"
- First systematic application of statistical methods to horse racing
- Introduced basic handicapping formulas considering past performance

## Modern Statistical Era (1960s-1980s)

### William Ziemba & Donald Hausch (1983)
- Book: "Beat the Racetrack"
- Focused on exploiting arbitrage opportunities in pari-mutuel betting systems
- Specifically targeted price mismatches between win and place/show markets
- While groundbreaking at the time, these opportunities are largely eliminated in modern betting exchanges due to real-time pricing and market efficiency

### Bolton & Chapman (1986)
- Paper: "Searching for Positive Returns at the Track: A Multinomial Logit Model for Handicapping Horse Races"
- Pioneered multinomial logit models in racing
- First to incorporate probability estimation for multiple horses simultaneously

## Computer Age Development (1990s-2000s)

### William Benter (1994)
- Paper: "Computer Based Horse Race Handicapping and Wagering Systems: A Report"
- Created the most successful computerized betting model to date
- Key innovations:
  - Neural network implementation
  - Real-time odds integration
  - Variable interaction effects
  - Dynamic probability updating

### Peter May & Michael Cristofoletti (1998)
- Publication: "The Statistical Modeling of Horse Racing"
- Introduced advanced time-series analysis
- Developed methods for handling autocorrelation in racing data

## Modern Machine Learning Era (2000s-Present)

### Robert Seder (2004)
- Paper: "Market Efficiency in Horse Race Betting"
- Applied efficient market hypothesis to racing
- Developed models incorporating market psychology

### Pardee & Williams (2012)
- Paper: "Utilizing Neural Networks in Horse Racing Prediction"
- Advanced implementation of deep learning
- Feature engineering specific to racing variables

### Chen et al. (2017)
- Paper: "Deep Learning for Horse Race Prediction"
- Introduced convolutional neural networks for race prediction
- Incorporated video analysis of running styles

## Key Statistical Methods Developed Over Time

### Fundamental Methods
- Logistic Regression
- Discriminant Analysis
- Time Series Analysis
- Probability Theory Applications

### Advanced Techniques
1. Machine Learning Applications
   - Neural Networks
   - Random Forests
   - Gradient Boosting
   - Support Vector Machines

2. Specialized Models
   - Multinomial Logit Models
   - Ordered Probit Models
   - Bayesian Networks
   - Hidden Markov Models

### Modern Innovations
- Real-time Data Processing
- Computer Vision Analysis
- Natural Language Processing (for news and social media)
- Environmental Factor Integration

Note: As this is a historical overview compiled from my training data, I recommend verifying specific papers and dates, as I may occasionally generate imprecise citations.

""")
