import numpy as np
from scipy.stats import norm

def black_scholes(S0, K, T, r, sigma):
    """
    Calculates the prices for a European call and put option using the
    Black-Scholes model.

    Args:
        S0 (float): Current stock price.
        K (float): Strike price.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.

    Returns:
        tuple: A tuple containing the call option price and the put option price.
    """
    # Calculate d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate call and put option prices
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    return call_price, put_price

# Example Usage
S0 = 100  # Current stock price
K = 100   # Strike price
T = 1     # Time to expiration (1 year)
r = 0.05  # Risk-free rate (5%)
sigma = 0.2 # Volatility (20%)

call, put = black_scholes(S0, K, T, r, sigma)

print(f"Call Option Price: ${call:.2f}")
print(f"Put Option Price: ${put:.2f}")