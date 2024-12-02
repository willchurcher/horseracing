import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")

def calculate_kelly_fraction(p, b):
    """Calculate the Kelly fraction for a simple bet"""
    return (p * b - (1-p)) / b

def calculate_growth_rate(f, p, b):
    """Calculate expected growth rate for a given betting fraction"""
    q = 1-p
    return p * np.log(1 + b*f) + q * np.log(1-f)

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
    logit_p = np.log(p/(1-p))
    pdf = np.exp(-(x - logit_p)**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    prob = 1/(1 + np.exp(-x))
    return np.trapezoid(prob * pdf, x)

def plot_uncertainty_effect(sigma):
    """Plot the effect of uncertainty on probability estimates"""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    p_values = np.concatenate(([0], np.linspace(0.01, 0.99, 98), [1]))
    ax.plot(p_values, p_values, 'k--', label='No uncertainty', alpha=0.5)
    
    expected_p = [calculate_expected_p(p, sigma) for p in p_values]
    ax.plot(p_values, expected_p, '-', label=f'σ = {sigma:.1f}')
    
    ax.set_xlabel('True probability (p)', fontsize=12)
    ax.set_ylabel('Expected probability E[P]', fontsize=12)
    ax.set_title('Effect of Uncertainty on Expected Probability', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

# Main Streamlit app
st.title("Kelly Criterion Analysis")

# Basic Kelly Formula section
st.markdown(r"""
### Basic Kelly Formula

If you know the probability of success, how much should you bet? Kelly staking provides the following formula:

$$
f^* = \frac{p \cdot d - 1}{d - 1}
$$
where 
- $f^*$ is the optimal fraction of your bankroll to bet
- $p$ is the probability of winning
- $d$ is the decimal odds. If you bet \$1, you receive \$$d$ if you win

This formula maximizes the expected growth rate of your bankroll, which can be explored in the plot below.
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
## Kelly Criterion Derivation

Let's derive the Kelly Criterion step by step. Our wealth after a single bet is:

$$
W_{n} = W_{n-1}(1 + f_n(X_nd - 1))
$$

where $f_n$ is the fraction bet, $X_n \sim Ber(p)$ is the outcome, and $d$ is the decimal odds.

Taking logarithms and summing over N bets:

$$
\log \frac{W_N}{W_0} = \sum_{n=1}^N \log(1 + f_n(X_nd - 1))
$$

As $N \to \infty$, by the Law of Large Numbers this converges to:

$$
\lim_{N \to \infty} \frac{1}{N} \log \frac{W_N}{W_0} = p\log(1 + f(d - 1)) + (1-p)\log(1 - f)
$$

Maximizing this expected growth rate yields the Kelly fraction:

$$
f^* = \frac{pd - 1}{d-1}
$$

## Uncertainty in Probability Estimates

In practice, we estimate probabilities using logistic regression:

$$
\log \frac{p_i}{1-p_i} = x_i^T\beta
$$

Maximum likelihood estimates have asymptotic normality:

$$
\hat{\beta} \sim N(\beta, I(\beta)^{-1})
$$

where the Fisher Information matrix for logistic regression is:

$$
I(\beta) = X^T\Lambda X
$$

with $\Lambda$ diagonal and $\Lambda_{ii} = p_i(1-p_i)$.

This leads to uncertainty in our probability estimates:

$$
\log \frac{P}{1-P} \sim N \left (\log \frac{p}{1-p},\sigma^2 \right )
$$

The Kelly fraction becomes:

$$
f^* = \frac{\mathbb{E}[P]d - 1}{d-1}
$$

The effect of this uncertainty is shown below:
""")

# Uncertainty effect plot
left_col, right_col = st.columns([1, 2])
with left_col:
    st.markdown("### Parameters")
    sigma = st.slider("Uncertainty (σ)", 0.0, 5.0, 2.0, step=0.1)
with right_col:
    st.pyplot(plot_uncertainty_effect(sigma))

# Favorite-longshot bias explanation
st.markdown("""
### The Favorite-Longshot Bias

The uncertainty adjustment naturally counteracts the favorite-longshot bias:
- Favorites (high probability events) have their expected probability reduced
- Longshots (low probability events) have their expected probability increased

This matches empirical observations in betting markets, where favorites tend to be underpriced and longshots overpriced.
""")