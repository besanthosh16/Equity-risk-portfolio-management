# Equity Risk Portfolio Management

A Python-based quantitative risk management platform for modeling, measuring, backtesting, and stress-testing market risk in equity portfolios. The platform implements Historical Simulation VaR, Expected Shortfall (ES), volatility-adjusted risk models, regulatory-style backtesting diagnostics, and interactive visualizations through a Streamlit dashboard.

## Overview

The platform evaluates portfolio downside risk under different market conditions using historical data and quantitative volatility models. It compares risk estimates across multiple methodologies and tests whether predicted losses align with realized market outcomes.

**Key capabilities:**

- Calculate 1-day and 10-day Value at Risk (VaR) and Expected Shortfall (ES/CVaR).
- Implement volatility scaling using EWMA, GARCH(1,1), and GJR-GARCH models.
- Backtest VaR forecasts using Kupiec and Christoffersen statistical tests.
- Classify backtesting results using the Basel Traffic Light framework.
- Run historical and hypothetical stress scenarios against custom portfolio weights.
- Visualize risk metrics, portfolio losses, volatility dynamics, and model performance through an interactive dashboard.

## Project Objectives

1. **Risk Measurement:** Calculate portfolio VaR and ES at 95%, 99%, and 99.5% confidence levels.
2. **Dynamic Volatility Adjustment:** Compare alternative volatility-scaling techniques to account for changing market conditions.
3. **Statistical Backtesting:** Evaluate VaR forecast accuracy using exception frequency, exception independence, and conditional coverage tests.
4. **Stress Testing:** Estimate portfolio losses under historical market crises and hypothetical market shocks.
5. **Model Benchmarking:** Compare risk estimates across models using exception rates, average VaR, ES, and forecast stability.
6. **Interactive Reporting:** Provide a dashboard for exploring portfolio risk, model diagnostics, and stress-test results.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10+ |
| Data Processing | pandas, NumPy |
| Scientific Computing | SciPy |
| Volatility Modeling | `arch`, `statsmodels` |
| Market Data | yfinance |
| Visualization | Plotly, Matplotlib |
| Interactive Dashboard | Streamlit |
| Testing | pytest |

## Project Architecture

```text
equity-risk-portfolio-management/
│
├── data/
│   └──                         # Cached raw and processed market data
│
├── src/
│   ├── data_pipeline.py        # Data fetching, cleaning, and return calculation
│   ├── portfolio.py            # Portfolio weights, returns, and daily P&L
│   ├── var_historical.py       # Historical Simulation VaR and Expected Shortfall
│   ├── volatility.py           # EWMA, GARCH, and GJR-GARCH models
│   ├── backtesting.py          # Kupiec, Christoffersen, and Basel diagnostics
│   ├── stress_testing.py       # Historical and hypothetical stress scenarios
│   └── charts.py               # Risk visualizations and distribution plots
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard
│
├── tests/
│   └── test_*.py               # Unit tests and validation checks
│
├── notebooks/
│   └──                         # Exploratory analysis and prototyping
│
├── requirements.txt            # Project dependencies
├── README.md
└── .gitignore
```

## Methodology

### 1. Market Data Collection and Preparation

Historical adjusted price data are collected using `yfinance`.

The data pipeline performs:

- Price alignment across assets and trading dates.
- Handling of missing observations.
- Adjusted-price processing for corporate actions, subject to data-provider coverage.
- Conversion of prices into daily logarithmic returns.

For asset \(i\), the daily log return is:

\[
r_{i,t}=\ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right)
\]

where \(P_{i,t}\) represents the adjusted closing price at time \(t\).

### 2. Portfolio Construction

The platform supports equal-weighted portfolios.

Portfolio returns are calculated as:

\[
r_{p,t}=\sum_{i=1}^{N}w_i r_{i,t}
\]

where \(w_i\) is the portfolio weight assigned to asset \(i\).

For a portfolio with a fixed initial capital \(V_0\), the daily profit and loss can be approximated as:

\[
P\&L_t=V_0r_{p,t}
\]

This approximation is suitable for small daily returns. A production implementation should account for changing portfolio value, rebalancing, transaction costs, and any applicable cash flows.

### 3. Historical Simulation VaR and Expected Shortfall

Historical Simulation estimates portfolio risk directly from observed historical returns without assuming that returns follow a normal distribution.

The platform evaluates rolling estimation windows of:

- 250 trading days
- 500 trading days
- 750 trading days

Risk estimates are produced at 95%, 99%, and 99.5% confidence levels.

For a loss variable \(L\), Value at Risk at confidence level \(\alpha\) is:

\[
VaR_{\alpha}=Q_{\alpha}(L)
\]

Expected Shortfall measures the average loss in the worst tail of the loss distribution:

\[
ES_{\alpha}=E[L\mid L\geq VaR_{\alpha}]
\]

For continuous loss distributions, this conditional expectation corresponds to the average loss beyond the VaR threshold. In empirical implementations, quantile conventions and treatment of observations equal to the threshold should be defined consistently.

**Interpretation:** At 99% confidence, a 1-day VaR of ₹10 lakh indicates that the modeled loss threshold is ₹10 lakh. Under the model, losses are expected to exceed this threshold approximately 1% of the time. It does not represent the maximum possible loss.

### 4. Volatility Scaling

Volatility scaling adjusts historical returns to reflect changes in current market volatility.

The general rescaling formula is:

\[
r_{t}^{*}=r_t\frac{\sigma_{\text{target}}}{\sigma_t}
\]

where:

- \(r_t\) is the historical return.
- \(\sigma_t\) is the estimated volatility associated with that return.
- \(\sigma_{\text{target}}\) is the target volatility for the forecast date.
- \(r_t^{*}\) is the rescaled return.

The implementation must use time-consistent volatility estimates and information available at each forecast date to avoid look-ahead bias.

#### EWMA — Exponentially Weighted Moving Average

EWMA gives greater weight to recent observations, allowing volatility estimates to react to changing market conditions.

The variance update is:

\[
\sigma_t^2=\lambda\sigma_{t-1}^2+
(1-\lambda)r_{t-1}^2
\]

The standard RiskMetrics decay factor is:

\[
\lambda=0.94
\]

for daily volatility estimation.

#### GARCH(1,1)

GARCH models volatility clustering, where periods of large market movements tend to be followed by further large movements.

\[
\sigma_t^2=\omega+\alpha r_{t-1}^2+
\beta\sigma_{t-1}^2
\]

The parameters capture the baseline variance, the impact of recent shocks, and the persistence of volatility.

#### GJR-GARCH

GJR-GARCH extends the standard GARCH model by incorporating asymmetric responses to negative returns.

\[
\sigma_t^2=\omega+\alpha r_{t-1}^2+
\gamma I(r_{t-1}<0)r_{t-1}^2+
\beta\sigma_{t-1}^2
\]

Here, \(I(r_{t-1}<0)\) equals 1 when the previous return is negative and 0 otherwise.

When \(\gamma>0\), negative returns increase conditional variance more than positive returns of the same magnitude, all else being equal.

### 5. VaR Backtesting

Backtesting compares forecast VaR thresholds with subsequently realized portfolio losses.

An exception occurs when the realized loss exceeds the predicted VaR threshold.

The platform evaluates:

**Kupiec Proportion of Failures (POF) Test**

Tests whether the observed exception frequency is statistically consistent with the expected exception probability.

**Christoffersen Independence Test**

Tests whether VaR exceptions occur independently or cluster in time.

**Christoffersen Conditional Coverage Test**

Combines exception frequency and independence to evaluate overall VaR coverage.

**Basel Traffic Light Framework**

Classifies exception counts into Green, Yellow, and Red zones for the applicable regulatory backtesting framework.

The precise thresholds depend on the prescribed observation window and regulatory methodology. The dashboard should clearly label the framework and sample size used. These diagnostics do not, by themselves, establish regulatory compliance.

### 6. Model Comparison

The platform benchmarks alternative risk models using:

| Metric | Purpose |
|---|---|
| Average VaR | Compares typical predicted loss thresholds |
| Exception Count | Measures how often realized losses breach VaR |
| Exception Rate | Compares observed breaches with the expected rate |
| Expected Shortfall | Evaluates the severity of modeled tail losses |
| Kupiec p-value | Assesses exception-frequency calibration |
| Independence p-value | Assesses exception clustering |
| Conditional Coverage p-value | Evaluates frequency and independence jointly |
| Forecast Stability | Examines how risk estimates change over time |

A model with fewer exceptions is not automatically better. VaR forecasts should be evaluated against the expected exception rate, statistical test results, tail-loss estimates, and forecast stability.

### 7. Stress Testing

Stress testing evaluates portfolio sensitivity to severe historical events and hypothetical market shocks.

**Historical scenarios may include:**

- COVID-19 market crash
- Other severe historical equity sell-offs supported by the available data

**Hypothetical scenarios may include:**

- Broad equity market declines
- Sector-specific shocks
- Individual asset price shocks
- Combined market-factor shocks

For portfolio weights \(w_i\) and scenario returns \(s_i\), the approximate scenario portfolio return is:

\[
r_{p,\text{stress}}=\sum_{i=1}^{N}w_i s_i
\]

The corresponding approximate loss for initial capital \(V_0\) is:

\[
L_{\text{stress}}=-V_0r_{p,\text{stress}}
\]

Positive values indicate losses; negative values indicate gains under the specified convention.

Historical stress results depend on scenario dates, asset availability, portfolio composition, and whether portfolio weights are held constant. Hypothetical shocks should be explicitly defined rather than presented as forecasts.

## Project Phases

| Phase | Description |
|---|---|
| 1 | Market data collection using financial APIs |
| 2 | Data cleaning, date alignment, and return calculation |
| 3 | Portfolio construction and daily P&L calculation |
| 4 | Historical Simulation VaR and Expected Shortfall |
| 5 | EWMA volatility scaling |
| 6 | GARCH(1,1) volatility scaling |
| 7 | GJR-GARCH asymmetric volatility modeling |
| 8 | VaR backtesting and statistical diagnostics |
| 9 | Risk-model comparison and benchmarking |
| 10 | Historical and hypothetical stress testing |

## Installation and Setup

### Prerequisites

- Python 3.10+ or higher
- Git
- Internet access for downloading market data

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd equity-risk-portfolio-management
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

Activate it on Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If `requirements.txt` is maintained:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, install the main dependencies directly:

```bash
pip install pandas numpy scipy yfinance arch statsmodels matplotlib plotly streamlit pytest
```

### 4. Run the test suite

```bash
pytest
```

The test suite should validate return calculations, portfolio aggregation, risk metrics, volatility models, and backtesting logic.

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Streamlit will provide a local URL for accessing the interactive dashboard.

## Dashboard Features

The dashboard is designed to provide the following functionality:

- **Portfolio Configuration:** Select equity tickers, portfolio weights, and the estimation window.
- **Risk Measurement:** Display 1-day and 10-day VaR and Expected Shortfall at supported confidence levels.
- **Volatility Models:** Compare Historical Simulation, EWMA-scaled, GARCH-scaled, and GJR-GARCH-scaled risk estimates.
- **Risk Visualization:** Plot rolling VaR against realized portfolio losses, volatility clustering, return distributions, and tail losses.
- **Backtesting Diagnostics:** Display exception counts, exception rates, p-values, and applicable Basel Traffic Light classifications.
- **Stress Analysis:** Present historical and hypothetical scenario losses and estimated capital at risk.
- **Model Benchmarking:** Compare model performance using consistent out-of-sample evaluation periods.

*Note: These are intended platform features. Their availability depends on the functionality implemented in the current codebase.*

## Expected Outputs

1. **Interactive Dashboard:** Configurable tickers, portfolio weights, estimation windows, confidence levels, and risk models.
2. **Risk Analytics:** VaR and ES estimates, return distributions, volatility charts, and rolling risk forecasts.
3. **Backtesting Reports:** Exception counts, breach rates, statistical test results, and applicable traffic-light classifications.
4. **Stress Loss Matrices:** Estimated portfolio losses under historical and hypothetical scenarios.
5. **Model Comparison Reports:** Side-by-side risk metrics and backtesting diagnostics.

## Implementation Notes and Limitations

### 1. Historical Quantiles

Historical Simulation does not require a normal-return assumption. However, it relies on the selected historical sample being informative about future risk. Short samples may underrepresent rare losses, while older observations may not reflect current market conditions.

### 2. 10-Day VaR

A common approximation is the square-root-of-time rule:

\[
VaR_{10}\approx VaR_1\sqrt{10}
\]

This approximation assumes suitable return aggregation behavior, including stable risk conditions and limited dependence. It is not generally exact for volatility-clustered returns, nonlinear portfolios, or extreme market regimes.

For greater consistency, the platform can estimate 10-day risk from aggregated 10-day portfolio returns or use an explicitly defined multi-day simulation method.

### 3. Volatility Rescaling

Volatility-adjusted historical simulation requires consistent scaling of historical shocks using volatility estimates available at the relevant forecast date. Incorrect alignment can introduce look-ahead bias and distort risk estimates.

### 4. Asymmetric Volatility

GJR-GARCH allows negative and positive shocks to have different effects on conditional variance. Its usefulness depends on parameter estimation, data quality, model assumptions, and the market regime.

### 5. Out-of-Sample Testing

Risk forecasts must be evaluated against future observations that were not used to estimate those forecasts. Rolling estimation windows and correctly aligned forecasts help prevent look-ahead bias.

### 6. Market Data Quality

Financial API data may contain missing observations, revised historical values, ticker changes, survivorship bias, and limitations in corporate-action adjustments. Results should be interpreted with these data limitations in mind.

### 7. Regulatory Interpretation

Statistical backtesting and Basel-style traffic-light classifications are analytical diagnostics. They do not independently establish regulatory approval, compliance, or suitability for institutional risk reporting.

## Future Enhancements

Potential extensions include:

- Monte Carlo VaR and Expected Shortfall.
- Filtered Historical Simulation.
- Automated model selection and parameter diagnostics.
- Correlation and covariance stress testing.
- Portfolio contribution-to-risk analysis.
- Transaction-cost and liquidity-risk adjustments.
- Exportable risk reports in CSV, Excel, or PDF formats.
- Automated testing and continuous integration.
- Deployment through a hosted Streamlit environment.

## Disclaimer

This project is intended solely for research and educational purposes. It does not constitute investment advice, financial planning, or a guarantee of regulatory compliance. VaR and Expected Shortfall are model-based estimates and cannot capture every possible market event. Historical performance and modeled risk estimates do not guarantee future outcomes.
