Equity risk portfolio management
Overview
A Python-based quantitative risk platform designed to model, scale, backtest, and stress-test market risk metrics for equity portfolios. The platform calculates Value at Risk (VaR) and Expected Shortfall (ES) across multiple volatility-adjustment frameworks, evaluates regulatory compliance via statistical backtesting, and presents analytics through an interactive dashboard.

Project Objective
1. Compute 1-day and 10-day Value at Risk (VaR) alongside Expected Shortfall (ES / CVaR).
2. Implement and compare volatility-scaling techniques to dynamically adjust historical returns to prevailing market volatility.
3. Formally back-test VaR models against regulatory statistical frameworks (Kupiec, Christoffersen, and Basel Traffic Light).
4. Perform historical and hypothetical scenario stress testing on custom portfolio weights.
5. Provide an interactive UI for risk visualization, diagnostic monitoring, and reporting.

Project Phases
Phase 1: Data Collection — Ingestion of multi-asset historical adjusted prices via financial APIs (yfinance).
Phase 2: Data Cleaning — Date alignment, handling splits/dividends, missing values, and generating stationary log-return matrices.
Phase 3: Portfolio Construction — Weight allocation (equal-weighted, market-cap, custom), daily portfolio returns, and tracking daily P&L.
Phase 4: Historical Simulation VaR — Rolling window empirical quantile calculation (250, 500, 750 days) across 95%, 99%, and 99.5% confidence levels.
Phase 5: EWMA Volatility Scaling — Exponentially Weighted Moving Average ($\lambda = 0.94$) filtering and return rescaling.
Phase 6: GARCH Volatility Scaling — Fitting GARCH(1,1) autoregressive conditional heteroskedasticity models for forward volatility scaling.
Phase 7: GJR-GARCH Volatility Scaling — Modeling leverage effects and asymmetric downside shocks.
Phase 8: VaR Backtesting — Rolling out-of-sample exception detection, Kupiec POF test, Christoffersen Independence & Conditional Coverage tests, and Basel traffic lighting.
Phase 9: Model Comparison — Side-by-side metric benchmarking across average VaR, exception rates, ES, and stability.
Phase 10: Stress Testing — Historical scenario playback (e.g., 2008 GFC, COVID-19 crash) and parametric factor shocks.

Technologies Used
Language: Python 3.10
Data & Numerical Computing: pandas, numpy, scipy
Time Series & Econometrics: arch (GARCH / GJR-GARCH), statsmodels
Market Data: yfinance
Visualization & UI: plotly, matplotlib, streamlit
Testing: pytest

Project Structure
equity-risk-portfolio-management/
├── data/                      # Cached raw & processed price data
├── src/
│   ├── data_pipeline.py       # Data fetching, date alignment & return math
│   ├── portfolio.py           # Portfolio construction & aggregate P&L
│   ├── var_historical.py      # Historical Simulation VaR & Expected Shortfall[cite: 1]
│   ├── volatility.py          # EWMA, GARCH(1,1), GJR-GARCH implementations[cite: 1]
│   ├── backtesting.py         # Kupiec, Christoffersen, & Basel tests[cite: 1]
│   ├── stress_testing.py      # Historical & hypothetical stress scenarios[cite: 1]
│   └── charts.py              # Visualizations & distribution plots[cite: 1]
├── dashboard/
│   └── app.py                 # Multi-page Streamlit application[cite: 1, 11]
├── tests/
│   └── test_*.py              # Unit tests for risk metrics & validation assertions[cite: 1]
├── notebooks/                 # Exploratory data analysis & prototyping[cite: 1]
├── requirements.txt           # Project dependencies
└── README.md

Setup
Prerequisites: Python 3.10 or higher & Git
Step-by-Step Installation
1. Create and activate a virtual environment
   source venv/bin/activate
2. Install dependencies
   pip install --upgrade pip
   pip install pandas numpy scipy yfinance arch statsmodels matplotlib plotly streamlit pytest
3. Verify the installation
   Run the test suite to ensure the data pipelines and VaR math pass all initial checks:Bashpytest
4. Launch the platform dashboard
   streamlit run dashboard/app.py

Outputs
1. Interactive Dashboard: Dynamic selectors for tickers, lookback windows (250/500/750d), confidence intervals (95%, 99%, 99.5%), and volatility models.
2. Statistical Visualizations: Rolling VaR vs. realized P&L overlays, volatility clustering charts, return distribution histograms, and tail losses.
3. Backtesting Diagnostic Reports: Summary tables displaying breach counts, fail ratios, $p$-values, and Basel Green/Yellow/Red zone classifications.
4. Stress Loss Matrices: Scenario breakdown showing projected drawdown and capital at risk under market shocks.

Methodology Notes
1. Empirical Quantiles: Historical simulation calculates VaR directly from historical percentiles without assuming standard normality.
2. Time Aggregation: 10-day VaR approximation via the $\sqrt{10}$ rule assumes identically and independently distributed (i.i.d.) returns; empirical volatility clustering can cause deviations from this assumption.
3. Volatility Rescaling: Returns are normalized by dynamic standard deviations:
      $$r_{t,\text{scaled}} = r_t \cdot \frac{\sigma_{\text{target}}}{\sigma_t}$$
      This prevents past calm regimes from underestimating risk and volatile spikes from over-inflating estimates.
5. Asymmetric Risk: GJR-GARCH incorporates a leverage term ($\gamma$) to account for negative market returns inducing higher volatility spikes than positive returns of identical magnitude.
6. Out-of-Sample Discipline: Back-testing utilizes strict rolling estimation windows to prevent look-ahead bias.

Disclaimer
This software is intended strictly for academic, research, and educational purposes. It does not constitute investment advice, financial planning, or regulatory risk assurance. Past market behavior and modeled estimates do not guarantee future performance.
