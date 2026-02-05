import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import statsmodels.api as sm

# DATA SIMULATION (SE3 vs SE4 Energy Dynamics)
def generate_energy_data(n=1000):
    np.random.seed(42)
    base = 20 + np.cumsum(np.random.normal(0, 0.2, n))
    
    se3 = 1.2 * base + np.random.normal(0, 1, n)
    
    beta_true = np.concatenate([np.full(n//2, 0.8), np.full(n//2, 1.1)])
    
    vol = 1.0
    eps = []
    for _ in range(n):
        vol = np.sqrt(0.05 + 0.1 * (vol**2 * 0.1) + 0.85 * vol**2)
        eps.append(np.random.standard_t(df=5) * vol)
        
    se4 = beta_true * se3 + 5 + np.array(eps)
    return pd.DataFrame({'SE3': se3, 'SE4': se4})

# KALMAN FILTER: DYNAMIC HEDGE RATIO ESTIMATION
def run_kalman_filter(y, x):
    """
    Estimates a time-varying Beta using a State-Space approach.
    State Equation: Beta_t = Beta_t-1 + Q
    Observation Equation: y_t = x_t * Beta_t + R
    """
    n = len(y)
    beta_hat = np.zeros(n)
    P = 1.0
    Q = 1e-5
    R = 0.1
    
    current_beta = 0.0
    
    for t in range(n):
        P = P + Q
        
        K = (P * x[t]) / (x[t]**2 * P + R)
        current_beta = current_beta + K * (y[t] - x[t] * current_beta)
        P = (1 - K * x[t]) * P
        
        beta_hat[t] = current_beta
        
    return beta_hat

# GARCH-T SIGNAL GENERATION
df = generate_energy_data()

df['beta_kf'] = run_kalman_filter(df['SE3'].values, df['SE4'].values)

df['spread'] = df['SE3'] - (df['beta_kf'] * df['SE4'])

garch = arch_model(df['spread'], vol='Garch', p=1, q=1, dist='t')
res = garch.fit(disp='off')
df['cond_vol'] = res.conditional_volatility

df['z_score'] = (df['spread'] - df['spread'].mean()) / df['cond_vol']

# BACKTEST & METRICS
df['signal'] = 0
df.loc[df['z_score'] > 2.0, 'signal'] = -1  # Short Spread
df.loc[df['z_score'] < -2.0, 'signal'] = 1  # Long Spread

# PnL Calculation
df['strategy_ret'] = df['spread'].diff() * df['signal'].shift(1)
sharpe = (df['strategy_ret'].mean() / df['strategy_ret'].std()) * np.sqrt(252)

print(f"Backtest Sharpe Ratio: {sharpe:.2f}")
print(f"Degrees of Freedom (Student's t): {res.params['nu']:.2f}")

# VISUALIZATION
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df['beta_kf'], label='Kalman Estimated Beta', color='darkblue')
ax1.set_title('Adaptive State Estimation (Kalman Filtered Hedge Ratio)')
ax1.legend()

ax2.plot(df['z_score'], label='GARCH-t Dynamic Z-Score', color='purple', alpha=0.6)
ax2.axhline(2, color='red', linestyle='--')
ax2.axhline(-2, color='green', linestyle='--')
ax2.set_title('Trading Signals: Volatility-Normalized Residuals')
ax2.legend()

plt.tight_layout()
plt.show()