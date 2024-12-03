import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")

def calculate_kelly_fraction(p, b):
    """Calculate the Kelly fraction for a simple bet"""
    return (p * b - (1 - p)) / b

def calculate_growth_rate(f, p, b):
    """Calculate expected growth rate for a given betting fraction"""
    q = 1 - p
    return p * np.log(1 + b * f) + q * np.log(1 - f)

def plot_growth_rate(p, b):
    """Plot growth rate vs betting fraction"""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    f = np.linspace(0, 1, 100)
    growth = [calculate_growth_rate(fi, p, b) for fi in f]
    kelly_f = calculate_kelly_fraction(p, b)
    
    ax.plot(f, growth)
    ax.axvline(x=kelly_f, color='r', linestyle='--')
    ax.set_xlabel('Betting Fraction', fontsize=12)
    ax.set_ylabel('Expected Growth Rate', fontsize=12)
    ax.set_title('Growth Rate vs Betting Fraction', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    
    return fig

def calculate_expected_p(p, sigma):
    """
    Calculate E[P] where logit(P) ~ N(logit(p), sigma^2)
    Uses numerical integration with special handling for p=0 and p=1
    """
    if p <= 0:
        return 0
    if p >= 1:
        return 1
    
    x = np.linspace(-50, 50, 1000)  # range for logit space
    logit_p = np.log(p / (1 - p))
    pdf = np.exp(-(x - logit_p)**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    prob = 1 / (1 + np.exp(-x))
    return np.trapz(prob * pdf, x)

def plot_uncertainty_effect(sigma):
    """Plot the effect of uncertainty on probability estimates"""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    p_values = np.concatenate(([0], np.linspace(0.01, 0.99, 98), [1]))
    ax.plot(p_values, p_values, 'k--', label='No uncertainty', alpha=0.5)
    
    expected_p = [calculate_expected_p(p, sigma) for p in p_values]
    ax.plot(p_values, expected_p, '-', label=f'σ = {sigma:.1f}')
    
    ax.set_xlabel('Original probability (p)', fontsize=12)
    ax.set_ylabel('Expected probability E[P]', fontsize=12)
    ax.set_title('Effect of Uncertainty on Expected Probability', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def compute_hat_matrix(X, p):
    """
    Compute the hat matrix for logistic regression
    
    Parameters:
    X: design matrix (n x p)
    p: vector of probabilities
    
    Returns:
    H: hat matrix (n x n)
    """
    # Create diagonal matrix of sqrt(p(1-p))
    L_inv = np.diag(1 / np.sqrt(p * (1 - p)))
    
    # Compute A = XL^(-1)
    A = X @ L_inv
    
    # Compute hat matrix
    H = A @ np.linalg.inv(A.T @ A) @ A.T
    
    return H

def compute_prediction_variance(x_new, X, p):
    """
    Compute variance for new prediction
    
    Parameters:
    x_new: new observation (p,)
    X: design matrix (n x p)
    p: vector of probabilities
    
    Returns:
    variance: scalar
    """
    L_inv = np.diag(1 / np.sqrt(p * (1 - p)))
    a_new = x_new @ L_inv
    A = X @ L_inv
    
    # Compute variance using hat matrix formulation
    variance = a_new @ np.linalg.inv(A.T @ A) @ a_new.T
    
    return variance

# Main Streamlit app
st.title("Kelly Criterion Analysis")

# Basic Kelly Formula section
st.markdown(r"""
### Basic Kelly Formula

The Kelly Criterion is a mathematical formula that determines the optimal size of a series of bets in order to maximize wealth. The formula is:

$$
f^* = \frac{p \cdot d - 1}{d - 1}
$$

where:
- \( f^* \) is the fraction of the current bankroll to bet
- \( p \) is the probability of winning
- \( d \) is the decimal odds.

This formula helps bettors maximize the expected growth rate of their bankroll.

""")

# Interactive Kelly plot section
left_col, right_col = st.columns([1, 2])
with left_col:
    st.markdown("### Parameters")
    p = st.slider("Win Probability", 0.0, 1.0, 0.6)
    b = st.slider("Decimal Odds", 1.0, 5.0, 2.0)
with right_col:
    st.pyplot(plot_growth_rate(p, b))

# Kelly Criterion Derivation
st.markdown(r"""
### Kelly Criterion Derivation

The Kelly Criterion is derived by maximizing the expected logarithmic growth rate of wealth. The derivation involves taking the logarithm of the wealth after a series of bets and maximizing the expected value.

The final result is the Kelly fraction:

$$
f^* = \frac{p \cdot d - 1}{d - 1}
$$

""")

# Accounting for estimation uncertainty
st.markdown(r"""
### Accounting for Estimation Uncertainty

In practice, probabilities are often estimated and subject to uncertainty. This uncertainty can be modeled using a logistic regression framework, where the log-odds of success are normally distributed.

The expected probability \( \mathbb{E}[P] \) can be calculated, and this expected probability is used in place of the point estimate \( p \) in the Kelly formula.

""")

# Uncertainty effect plot
left_col, right_col = st.columns([1, 2])
with left_col:
    st.markdown("### Parameters")
    sigma = st.slider("Uncertainty (σ)", 0.0, 5.0, 2.0, step=0.1)
with right_col:
    st.pyplot(plot_uncertainty_effect(sigma))

# Favorite-longshot bias explanation
st.markdown(r"""
### The Favorite-Longshot Bias

The favorite-longshot bias refers to the phenomenon where favorites are underbet and longshots are overbet. The Kelly Criterion, when adjusted for uncertainty, naturally counteracts this bias by reducing the expected probability of favorites and increasing that of longshots.
""")

# General setting and further questions
st.markdown(r"""
### The General Setting

In a more complex betting environment, such as a pool where the house takes a percentage of winnings, the Kelly Criterion can be extended to account for these additional factors. The wealth after a bet is given by:

$$
W_n = W_{n-1} - \sum_h B_h + \sum_h B_h \cdot (1 - R) \cdot \frac{\sum_j B_j + M_j}{B_h + M_h} \cdot X_h
$$

The goal is to maximize the expected logarithmic growth rate of wealth.

### Further Questions

- **Is Kelly truly what should be optimized?**
  - The Kelly Criterion is a good starting point, but fractional Kelly or other risk-adjusted strategies might be more appropriate in practice.
  
- **How does bias impact bet sizing?**
  - Bias in probability estimates can lead to suboptimal bet sizes. Accounting for model uncertainty may help mitigate this, but further analysis is needed.
  
- **Optimizing the final objective in practice:**
  - Stochastic optimization methods could be used, but ensuring non-negativity of bets and handling the complexity of the objective function would be challenging.
  
- **Connections with reinforcement learning:**
  - Reinforcement learning, particularly algorithms like Proximal Policy Optimization, could be useful for dynamically adjusting bet sizes based on past outcomes.
  
- **Suitable models:**
  - Models that can handle uncertainty and provide probabilistic predictions, such as Bayesian logistic regression, might be suitable for this setting.


### My Thoughts

The Kelly Criterion is a powerful tool for bettors and investors looking to optimize their wealth growth. However, in real-world scenarios, the assumptions of the Kelly Criterion are often violated, particularly the assumption of known probabilities. By accounting for uncertainty in probability estimates, the Kelly Criterion can be made more robust, but there is still much to explore in terms of practical implementation.

The favorite-longshot bias is an interesting phenomenon that the Kelly Criterion, when adjusted for uncertainty, can help mitigate. However, in more complex betting environments, such as those involving house take rates or multiple outcomes, the Kelly Criterion must be extended and adapted to account for these additional factors.

The questions raised at the end of the application highlight the complexity of applying the Kelly Criterion in practice. While the Kelly Criterion provides a solid theoretical foundation, practical considerations such as model bias, optimization challenges, and connections with reinforcement learning must be addressed to fully realize its potential.

Overall, this application provides a solid introduction to the Kelly Criterion and its extensions, but there is still much work to be done in terms of practical implementation and further research.

""")