# energy-stat-arb-kalman-garch

# Adaptive Energy Spread Trading: A Kalman-GARCH Hybrid

## Summary
This project implements a pairs trading strategy targeting locational price discrepancies (e.g., SE3 vs SE4) in the Nordic Power Market. By merging **State-Space modeling** with **Conditional Heteroskedasticity frameworks**, the model successfully isolates tradable alpha from the high-noise, non-stationary environment of hourly electricity prices.

## Core Methodology

### 1. Dynamic Regime Adaptation (Kalman Filter)
Traditional OLS-based hedge ratios fail during structural market shifts, for example, as sudden grid maintenance or changes in interconnector flows. 
* **State Estimation:** This engine utilizes a Kalman Filter to treat the hedge ratio ($\beta$) as a latent state variable. 
* **Recursive Updates:** The model recursively updates its belief of "Fair Value" as new price data arrives, allowing the strategy to pivot during structural breaks.
* **Lag Reduction:** This eliminates the "lag" inherent in rolling-window OLS and significantly reduces drawdowns during regime shifts.



### 2. Volatility-Normalized Execution (GARCH-t)
Energy markets exhibit extreme volatility clustering and heavy tails that violate standard normal distribution assumptions.
* **GARCH(1,1) Integration:** The model estimates time-varying conditional variance ($\sigma_t^2$) on the Kalman residuals.
* **Student's t-Distribution:** By fitting residuals to a Student's t-distribution ($\nu \approx 6.5$), the strategy accounts for "fat-tail" events, preventing premature entries during expansionary volatility phases.
* **The Signal:** Entry and exit triggers are defined by a Dynamic Z-Score, where the spread is normalized by the GARCH-predicted volatility.

## Performance Highlights
* **Annualized Sharpe Ratio:** 3.13
* **Robustness:** Demonstrated ability to remain market-neutral across diverse volatility regimes.
* **Risk Management:** Student's t-parameters confirm a significant reduction in "Kurtosis Risk" compared to Gaussian models.




*Note: This model is for research purposes and utilizes simulated data to demonstrate the mathematical framework.*
