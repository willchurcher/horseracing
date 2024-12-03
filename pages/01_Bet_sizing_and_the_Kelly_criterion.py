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

# Add these functions to the top of your file

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
    L_inv = np.diag(1 / np.sqrt(p * (1-p)))
    
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
    L_inv = np.diag(1 / np.sqrt(p * (1-p)))
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

## Accounting for estimation uncertainty

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

For an out of sample observation x:

$$
\log\frac{P}{1-P} = x^T \hat \beta ~ \sim N \left (x^T\beta,x^T(X^T\Lambda X)^{-1}x \right)
$$

As a result, the distribution of the probability is itself random, which is effectively a Bayesian insight.

## Consequences of random probabilities

How does the derivation of the Kelly criterion change if $P$ is non-deterministic? If one follows the original argument, they will find all that changes in the result is the use
of $\mathbb{E}[P]$ in place of $p$, giving the new Kelly fraction of:
$$
f^* = \frac{\mathbb{E}[P]d - 1}{d-1}
$$

How does $\mathbb{E}[P]$ diverge from $p$ as variance increases? Let:
$$
\log \frac{P}{1-P} \sim N \left (\log \frac{p}{1-p},\sigma^2 \right )
$$

The chart below demonstrates the impact of increasing variance on expected probability.

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
## The Favorite-Longshot Bias

The uncertainty adjustment naturally counteracts the favorite-longshot bias:
- Favorites (high probability events) have their expected probability reduced.
- Longshots (low probability events) have their expected probability increased.

""")



st.markdown(r'''
## The general setting
Suppose now you are betting in a pool where the house takes $R\cdot 100\%$ of the winnings. Suppose:
- The amount the market bets on horse $h$ is $M_h$. 
- Horse $h$ winning is represented with a Bernoulli random variable $X_h$.

Each of these quantities will be modelled. What is the distribution of the expected 

$$
W_n = W_{n-1} - \sum_h B_h + \sum_h B_h\cdot (1-R)\cdot \frac{\sum_j B_j+M_j}{B_h+M_h}\cdot X_h
$$

$$
\log \frac{W_n}{W_{n-1}} = \log \left(1 + \sum_h \frac{B_h}{W_{n-1}} \cdot \left ((1-R)\cdot \frac{\sum_j B_j + M_j}{B_h+M_h}\cdot X_h -1 \right )\right) 
$$

By the logic of Kelly, we aim to maximise this growth rate in expectation:

$$
f(B_{1:H}) = \mathbb{E} \left [\log \left(1 + \sum_{h=1}^H \frac{B_h}{W_{n-1}} \cdot \left ((1-R)\cdot \frac{\sum_j B_j + M_j}{B_h+M_h}\cdot X_h -1 \right )\right) \right ]
$$

Assuming we can sample from this posterior distribution, then stochastic optimisation methods can be used to determine the optimal bet sizes $B_h$.

### Conclusions

- The market sizes $M_h$ and outcome indicators $X_h$ should be modelled jointly.
- There is a natural relationship between $B_h$ and $W_{n-1}$ as well as between $B_h$ and $M_h$.
    - Trading off these terms will lead to greater execution performance.

## Further questions

- Is Kelly truly what should be optimised?
    - What about fractional Kelly?
    - Or optimising some risk-adjusted Kelly?
- So far we have accounted for parameter uncertainty in the model. 
    - How would bias impact the bet sizing? 
    - Or does accounting for a model's uncertainty handle this naturally?
- How would the final objective be optimised in practice?
    - Would penalisation and regularisation terms be of use?
    - How would you ensure bets are non-negative?
    - Would an Adam type stochastic optimiser work?
    - Is the gradient of the function inside the expectation an unbiased estimator of the gradient of the objective function?
        - In this case stochastic gradient descent would be useful.
- What are the ties with reinforcement learning?
    - Proximal Policy Optimisation feels appropriate somehow...
- What models are suitable here?

''')