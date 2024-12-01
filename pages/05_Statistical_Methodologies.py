import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_kelly_fraction(p, b):
    """Calculate the Kelly fraction for a simple bet"""
    return (p * b - (1-p)) / b

def calculate_growth_rate(f, p, b):
    """Calculate expected growth rate for a given betting fraction"""
    q = 1-p
    return p * np.log(1 + b*f) + q * np.log(1-f)

def plot_growth_rate(p, b):
    """Plot growth rate vs betting fraction"""
    # Set figure size and DPI for better readability
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    f = np.linspace(0, 1, 100)
    growth = [calculate_growth_rate(fi, p, b) for fi in f]
    kelly_f = calculate_kelly_fraction(p, b)
    
    ax.plot(f, growth)
    ax.axvline(x=kelly_f, color='r', linestyle='--')
    
    # Increase font sizes
    ax.set_xlabel('Betting Fraction', fontsize=12)
    ax.set_ylabel('Expected Growth Rate', fontsize=12)
    ax.set_title('Growth Rate vs Betting Fraction', fontsize=14)
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig

st.title("Kelly Criterion Analysis")

st.markdown(r"""
### Complete Kelly Criterion Derivation

If you know the probability of success, how much should you bet? Kelly staking provides the following formula:

$$
f^* = \frac{p \cdot d - 1}{d - 1}
$$
where 
- $p$ is the probability of winning.
- $d$ is the decimal odds. If you bet \$1, you receive \$$d$ if you win.

This formula is chosen to maximize the expected growth rate of your bankroll, which can be explored in the plot below.

## Wealth growth interactive plot

""")


# Create two columns for the entire layout
left_col, right_col = st.columns([1, 2])

# Put controls in left column
with left_col:
    st.markdown("### Parameters")
    p = st.slider("Win Probability", 0.0, 1.0, 0.6)
    b = st.slider("Decimal Odds", 1.0, 5.0, 2.0)

# Put plot in right column
with right_col:
    st.pyplot(plot_growth_rate(p, b))

st.markdown(r"""
## Complete Kelly Criterion Derivation

Let's derive the Kelly Criterion step by step, starting with a single bet and extending to the long-term optimal betting fraction.
If our wealth at time $n$ is $W_n$, then our wealth at time $n+1$ will be:
$$
W_{n} = W_{n-1}(1-f_n) + W_{n-1}f_n \cdot X_n \cdot d
$$
where $f_n$ is the fraction of wealth bet at time $n$, $X_n \sim Ber(p)$ represents the outcome of the bet occuring with probability $p$, and $d$ represents the decimal odds.

Rearranging:
$$
\frac{W_{n}}{W_{n-1}} = 1-f_n + f_nX_nd = 1 + f_n(X_nd - 1)
$$
Taking logarithms to make multiplication additive:
$$
\log W_{n} - \log W_{n-1} = \log(1 + f_n(X_nd - 1))
$$
For N bets, we can write this as a sum:
$$
\frac{1}{N} \log W_N - \log W_0 = \frac{1}{N} \log \frac{W_N}{W_0} = \frac{1}{N} \sum_{n=1}^N \log(1 + f_n(X_nd - 1))
$$
By the Law of Large Numbers, as N approaches infinity, this converges to the expected value:
$$
\lim_{N \to \infty} \frac{1}{N} \log \frac{W_N}{W_0} = \mathbb{E}[\log(1 + f(Xd - 1))]
$$
$$
= p\log(1 + f(d - 1)) + (1-p)\log(1 - f)
$$
To find the optimal fraction $f^*$, we differentiate with respect to $f$ and set equal to zero:
$$
\frac{\partial}{\partial f}E[\log(1 + f(Xd - 1))] = \frac{p(d-1)}{1 + f(d-1)} - \frac{1-p}{1-f} = 0
$$
Solving this equation gives the Kelly fraction:
$$
f^* = \frac{pd - 1}{d-1} = \frac{p - \frac{1}{d}}{1 - \frac{1}{d}}
$$
""")


st.markdown(r""" 
## Imperfect worlds

Crucial to the derviation, is the knowledge of the true win probability, $p$. In practice, a model is constructed which will combine estimated parameters with quantities which
are observed before the race begins. One common model is the **logistic model** (multinomial model in general), which models success probability as follows:

$$
\log \frac{p_i}{1-p_i} = \beta\cdot x_i
$$

where $x_i$ is a vector of observed quantities. This process is then fit using maximum likelihood estimation.

It is well known that maximum likelihood estimates $\hat{\beta}$ for the $\beta$ parameter converge to a normal distribution:

$$
\hat{\beta} \overset{d}{\to} N(\beta, I(\beta)^{-1})
$$

where $I(\beta)$ is the Fisher information matrix.

### How does this affect the Kelly Criterion?

We now know that $p$ is a random variable - how does this affect the Kelly Criterion? Replacing $p$ with a random variable $P$, the only changes to our formula are that we now
have an expected value of $p$ in our Kelly Criterion formula:

$$
f^* = \frac{\mathbb{E}[P]d - 1}{d-1}
$$

This E[P] can now be calcualted using the Fisher information matrix and the data we have collected. To avoid using data, we can model the distribution of $P$ such that 

$$
\log \frac{P}{1-P} \sim N \left (\log \frac{p}{1-p},\sigma^2 \right )
$$

This has the effect of shrinking the Kelly Criterion towards 0.5, which can be seen in the plot below.


""")

def calculate_expected_p(p, sigma):
    """
    Calculate E[P] where logit(P) ~ N(logit(p), sigma^2)
    Uses numerical integration with special handling for p=0 and p=1
    """
    # Handle edge cases
    if p <= 0:
        return 0
    if p >= 1:
        return 1
    
    # Use numerical integration to find E[P]
    x = np.linspace(-50, 50, 1000)  # range for logit space
    logit_p = np.log(p/(1-p))
    # Normal pdf centered at logit_p with variance sigma^2
    pdf = np.exp(-(x - logit_p)**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    # Transform back to probability space
    prob = 1/(1 + np.exp(-x))
    return np.trapezoid(prob * pdf, x)

# Create two columns for the layout
left_col, right_col = st.columns([1, 2])

# Put control in left column
with left_col:
    st.markdown("### Parameters")
    sigma = st.slider("Uncertainty (σ)", 0.0, 3.0, 1.0, step=0.1)

# Put plot in right column
with right_col:
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    
    # Create range of true probabilities including endpoints
    p_values = np.concatenate(([0], np.linspace(0.01, 0.99, 98), [1]))
    
    # Plot diagonal line for reference (no uncertainty)
    ax.plot(p_values, p_values, 'k--', label='No uncertainty', alpha=0.5)
    
    # Plot expected probabilities for chosen sigma
    expected_p = [calculate_expected_p(p, sigma) for p in p_values]
    ax.plot(p_values, expected_p, '-', label=f'σ = {sigma:.1f}')
    
    # Formatting
    ax.set_xlabel('True probability (p)', fontsize=12)
    ax.set_ylabel('Expected probability E[P]', fontsize=12)
    ax.set_title('Effect of Uncertainty on Expected Probability', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Make plot square
    ax.set_aspect('equal')
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    st.pyplot(fig)