import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import chi2
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Market Risk Model Dashboard",
    layout="wide",
    page_icon="📈",
)


# ---------------------------------------------------------
# Helper Functions & Caching
# ---------------------------------------------------------
@st.cache_data
def load_clean_prices():
  if os.path.exists("clean_price_matrix(10).csv"):
    df = pd.read_csv(
        "clean_price_matrix(10).csv", parse_dates=["Date"], index_col="Date"
    )
    return df
  else:
    # Fallback simulation if clean_price_matrix(10).csv is missing
    dates = pd.date_range("2016-07-29", "2026-07-28", freq="B")
    dow_10 = [
        "GS",
        "CAT",
        "UNH",
        "TRV",
        "MSFT",
        "AMGN",
        "V",
        "JPM",
        "AAPL",
        "HD",
    ]
    np.random.seed(42)
    rets = pd.DataFrame(
        np.random.normal(0.0004, 0.012, (len(dates), len(dow_10))),
        index=dates,
        columns=dow_10,
    )
    prices = (1 + rets).cumprod() * 100
    return prices


def simulate_portfolio_series(prices, weights, initial_val):
  norm_w = weights / weights.sum()
  log_rets = np.log(prices / prices.shift(1)).dropna()
  port_rets = log_rets.dot(norm_w)
  port_val = initial_val * np.exp(port_rets.cumsum())
  port_val = pd.Series(
      [initial_val] + port_val.tolist(),
      index=[prices.index[0]] + list(port_rets.index),
  )
  daily_pnl = port_val.diff().fillna(0.0)
  return port_rets, port_val, daily_pnl


def fit_garch_scipy(returns):
  scale = 100.0
  r = returns.values * scale
  mu = np.mean(r)
  eps = r - mu
  var_eps = np.var(eps)

  def neg_log_lik(params):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
      return 1e10
    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[0] = var_eps
    for t in range(1, n):
      sigma2[t] = omega + alpha * (eps[t - 1] ** 2) + beta * sigma2[t - 1]
      if sigma2[t] <= 0:
        sigma2[t] = 1e-6
    return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + (eps**2) / sigma2)

  bounds = [(1e-6, 1.0), (1e-6, 0.49), (1e-6, 0.99)]
  constraints = {"type": "ineq", "fun": lambda p: 0.999 - (p[1] + p[2])}
  init_p = [var_eps * 0.05, 0.08, 0.88]
  res = minimize(
      neg_log_lik,
      init_p,
      method="SLSQP",
      bounds=bounds,
      constraints=constraints,
  )
  omega_s, alpha_g, beta_g = res.x
  n = len(eps)
  sigma2_s = np.zeros(n)
  sigma2_s[0] = var_eps
  for t in range(1, n):
    sigma2_s[t] = omega_s + alpha_g * (eps[t - 1] ** 2) + beta_g * sigma2_s[t - 1]
  omega = omega_s / (scale**2)
  cond_vol = pd.Series(np.sqrt(sigma2_s) / scale, index=returns.index)
  eps_orig = returns.values - (mu / scale)
  forecast_var = omega + alpha_g * (eps_orig**2) + beta_g * (cond_vol.values**2)
  forecast_vol = pd.Series(np.sqrt(forecast_var), index=returns.index)
  return omega, alpha_g, beta_g, cond_vol, forecast_vol


def fit_gjr_garch_scipy(returns):
  scale = 100.0
  r = returns.values * scale
  mu = np.mean(r)
  eps = r - mu
  var_eps = np.var(eps)

  def neg_log_lik(params):
    omega, alpha, gamma, beta = params
    if (
        omega <= 0
        or alpha < 0
        or gamma < 0
        or beta < 0
        or (alpha + 0.5 * gamma + beta) >= 1.0
    ):
      return 1e10
    n = len(eps)
    sigma2 = np.zeros(n)
    sigma2[0] = var_eps
    for t in range(1, n):
      e_prev = eps[t - 1]
      s_prev = sigma2[t - 1]
      ind = 1.0 if e_prev < 0 else 0.0
      sigma2[t] = (
          omega
          + alpha * (e_prev**2)
          + gamma * ind * (e_prev**2)
          + beta * s_prev
      )
      if sigma2[t] <= 0:
        sigma2[t] = 1e-6
    return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + (eps**2) / sigma2)

  bounds = [(1e-6, 1.0), (1e-6, 0.49), (1e-6, 0.49), (1e-6, 0.99)]
  constraints = {
      "type": "ineq",
      "fun": lambda p: 0.999 - (p[1] + 0.5 * p[2] + p[3]),
  }
  init_p = [var_eps * 0.05, 0.05, 0.10, 0.80]
  res = minimize(
      neg_log_lik,
      init_p,
      method="SLSQP",
      bounds=bounds,
      constraints=constraints,
  )
  omega_s, alpha_g, gamma_g, beta_g = res.x
  n = len(eps)
  sigma2_s = np.zeros(n)
  sigma2_s[0] = var_eps
  for t in range(1, n):
    e_prev = eps[t - 1]
    s_prev = sigma2_s[t - 1]
    ind = 1.0 if e_prev < 0 else 0.0
    sigma2_s[t] = (
        omega_s
        + alpha_g * (e_prev**2)
        + gamma_g * ind * (e_prev**2)
        + beta_g * s_prev
    )
  omega = omega_s / (scale**2)
  cond_vol = pd.Series(np.sqrt(sigma2_s) / scale, index=returns.index)
  eps_orig = returns.values - (mu / scale)
  ind_t = (eps_orig < 0).astype(float)
  forecast_var = (
      omega
      + alpha_g * (eps_orig**2)
      + gamma_g * ind_t * (eps_orig**2)
      + beta_g * (cond_vol.values**2)
  )
  forecast_vol = pd.Series(np.sqrt(forecast_var), index=returns.index)
  return omega, alpha_g, gamma_g, beta_g, cond_vol, forecast_vol


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("📊 Risk Analytics Platform")
page = st.sidebar.radio(
    "Navigate Pages",
    [
        "1. Portfolio Setup",
        "2. Value at Risk (VaR)",
        "3. Volatility Scaling",
        "4. Backtesting Engine",
        "5. Stress Testing",
    ],
)

# Load Base Data
all_prices = load_clean_prices()
dow_tickers = all_prices.columns.tolist()

# ---------------------------------------------------------
# PAGE 1: PORTFOLIO SETUP
# ---------------------------------------------------------
if page == "1. Portfolio Setup":
  st.title("💼 Portfolio Configuration & Analytics")

  col1, col2 = st.columns([1, 2])
  with col1:
    st.subheader("Configuration Inputs")
    selected_tickers = st.multiselect(
        "Select Tickers", dow_tickers, default=dow_tickers[:10]
    )
    initial_val = st.number_input(
        "Initial Portfolio Value ($)",
        min_value=1000.0,
        value=100000.0,
        step=10000.0,
    )

    weight_mode = st.radio(
        "Weighting Scheme", ["Equal-Weighted", "Custom Weights"]
    )
    if weight_mode == "Equal-Weighted":
      weights = (
          np.repeat(1.0 / len(selected_tickers), len(selected_tickers))
          if selected_tickers
          else np.array([])
      )
    else:
      raw_w = []
      for t in selected_tickers:
        raw_w.append(
            st.number_input(
                f"Weight for {t}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(selected_tickers),
                step=0.01,
            )
        )
      weights = np.array(raw_w)

    min_d, max_d = (
        all_prices.index.min().date(),
        all_prices.index.max().date(),
    )
    date_range = st.date_input(
        "Date Range", [min_d, max_d], min_value=min_d, max_value=max_d
    )

  with col2:
    if len(selected_tickers) > 0 and len(date_range) == 2:
      start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(
          date_range[1]
      )
      filtered_prices = all_prices.loc[start_date:end_date, selected_tickers]
      port_rets, port_val, daily_pnl = simulate_portfolio_series(
          filtered_prices, weights, initial_val
      )

      # Key Metric Cards
      m1, m2, m3, m4 = st.columns(4)
      m1.metric("Final Value", f"${port_val.iloc[-1]:,.2f}")
      m2.metric("Total Return", f"{(port_val.iloc[-1]/initial_val - 1):.2%}")
      m3.metric("Annualized Vol", f"{port_rets.std() * np.sqrt(252):.2%}")
      m4.metric(
          "Max Drawdown", f"{((port_val / port_val.cummax()) - 1).min():.2%}"
      )

      # Interactive Chart
      fig = go.Figure()
      fig.add_trace(
          go.Scatter(
              x=port_val.index,
              y=port_val,
              mode="lines",
              name="Portfolio Value ($)",
              line=dict(color="#1f77b4", width=2),
          )
      )
      fig.update_layout(
          title="Portfolio Value Over Time",
          xaxis_title="Date",
          yaxis_title="Portfolio Value ($)",
          template="plotly_white",
      )
      st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: VALUE AT RISK (VaR)
# ---------------------------------------------------------
elif page == "2. Value at Risk (VaR)":
  st.title("🛡️ Value at Risk (VaR) & Expected Shortfall")

  port_rets, port_val, daily_pnl = simulate_portfolio_series(
      all_prices, np.repeat(1.0 / len(dow_tickers), len(dow_tickers)), 100000.0
  )

  col1, col2 = st.columns([1, 2])
  with col1:
    st.subheader("Model Parameters")
    conf_level = st.selectbox(
        "Confidence Level", [0.95, 0.99, 0.995], index=1
    )
    window = st.slider(
        "Lookback Window (Days)",
        min_value=100,
        max_value=750,
        value=250,
        step=50,
    )
    var_model = st.selectbox(
        "VaR Model",
        [
            "Historical Simulation",
            "EWMA Volatility-Scaled",
            "GARCH(1,1)-Scaled",
            "GJR-GARCH-Scaled",
        ],
    )

  with col2:
    alpha = 1.0 - conf_level
    rolling_var = pd.Series(index=daily_pnl.index, dtype=float)
    rolling_es = pd.Series(index=daily_pnl.index, dtype=float)

    for i in range(window, len(daily_pnl)):
      win_pnl = daily_pnl.iloc[i - window : i]
      v = -np.percentile(win_pnl, alpha * 100)
      rolling_var.iloc[i] = v
      tail = win_pnl[win_pnl <= -v]
      rolling_es.iloc[i] = -tail.mean() if len(tail) > 0 else v

    m1, m2, m3 = st.columns(3)
    m1.metric(
        f"Current {conf_level*100}% VaR",
        f"${rolling_var.dropna().iloc[-1]:,.2f}",
    )
    m2.metric(
        f"Current {conf_level*100}% Expected Shortfall",
        f"${rolling_es.dropna().iloc[-1]:,.2f}",
    )
    m3.metric(
        "Avg Daily Loss Tail", f"${rolling_es.dropna().mean():,.2f}"
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rolling_var.index,
            y=rolling_var,
            mode="lines",
            name=f"{conf_level*100}% VaR ($)",
            line=dict(color="orange", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_es.index,
            y=rolling_es,
            mode="lines",
            name=f"{conf_level*100}% Expected Shortfall ($)",
            line=dict(color="darkred", dash="dot"),
        )
    )
    fig.update_layout(
        title=(
            f"Rolling {conf_level*100}% VaR and Expected Shortfall"
            f" ({window}-Day Window)"
        ),
        xaxis_title="Date",
        yaxis_title="Loss Magnitude ($)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: VOLATILITY SCALING
# ---------------------------------------------------------
elif page == "3. Volatility Scaling":
  st.title("⚡ Dynamic Volatility Modeling & Scaling")

  port_rets, port_val, daily_pnl = simulate_portfolio_series(
      all_prices, np.repeat(1.0 / len(dow_tickers), len(dow_tickers)), 100000.0
  )

  col1, col2 = st.columns([1, 2])
  with col1:
    st.subheader("Model Settings")
    ewma_lam = st.slider(
        "EWMA Decay Factor (λ)",
        min_value=0.85,
        max_value=0.99,
        value=0.94,
        step=0.01,
    )

    # Fit models
    sq_rets = (port_rets.shift(1) ** 2).fillna(port_rets.var())
    ewma_vol = np.sqrt(sq_rets.ewm(alpha=1 - ewma_lam, adjust=False).mean())

    omega_g, alpha_g, beta_g, garch_vol, _ = fit_garch_scipy(port_rets)
    omega_j, alpha_j, gamma_j, beta_j, gjr_vol, _ = fit_gjr_garch_scipy(
        port_rets
    )

    st.markdown("**Fitted Parameters:**")
    st.write(
        f"• **GARCH(1,1):** $\\alpha={alpha_g:.4f}, \\beta={beta_g:.4f}$"
    )
    st.write(f"• **GJR-GARCH:** $\\gamma={gamma_j:.4f}$ (Leverage Shock)")

  with col2:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ewma_vol.index,
            y=ewma_vol * np.sqrt(252),
            name=f"EWMA (λ={ewma_lam})",
            line=dict(color="purple", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=garch_vol.index,
            y=garch_vol * np.sqrt(252),
            name="GARCH(1,1)",
            line=dict(color="green", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gjr_vol.index,
            y=gjr_vol * np.sqrt(252),
            name="GJR-GARCH (Asymmetric)",
            line=dict(color="crimson", width=1.5),
        )
    )
    fig.update_layout(
        title="Annualized Volatility Trajectory Comparison",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 4: BACKTESTING ENGINE
# ---------------------------------------------------------
elif page == "4. Backtesting Engine":
  st.title("🎯 Statistical VaR Backtesting & Basel Traffic Light")

  port_rets, port_val, daily_pnl = simulate_portfolio_series(
      all_prices, np.repeat(1.0 / len(dow_tickers), len(dow_tickers)), 100000.0
  )
  actual_losses = -daily_pnl

  window = 250
  alpha = 0.01
  raw_var = pd.Series(index=daily_pnl.index, dtype=float)
  for i in range(window, len(daily_pnl)):
    raw_var.iloc[i] = -np.percentile(daily_pnl.iloc[i - window : i], alpha * 100)

  valid_mask = raw_var.notna()
  eval_losses = actual_losses[valid_mask]
  eval_var = raw_var[valid_mask]
  exceptions = (eval_losses > eval_var).astype(int)
  x = exceptions.sum()
  N = len(eval_losses)
  p_hat = x / N

  # Kupiec POF
  p0 = 0.01
  log_num = (N - x) * np.log(1 - p0) + x * np.log(p0)
  log_den = (N - x) * np.log(1 - p_hat) + x * np.log(p_hat)
  lr_pof = -2 * (log_num - log_den)
  p_val_pof = 1.0 - chi2.cdf(lr_pof, df=1)

  # Basel Zone
  scaled_x = (x / N) * 250
  zone = (
      "Green (Pass)"
      if scaled_x <= 4.99
      else "Yellow (Monitor)"
      if scaled_x <= 9.99
      else "Red (Reject)"
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Exceptions", f"{x} / {N}")
  col2.metric("Exception Ratio", f"{p_hat:.2%}")
  col3.metric("Kupiec POF P-Value", f"{p_val_pof:.4f}")
  col4.metric("Basel Zone", zone)

  # Interactive P&L vs VaR plot
  fig = go.Figure()
  fig.add_trace(
      go.Scatter(
          x=eval_losses.index,
          y=eval_losses,
          mode="lines",
          name="Actual Loss ($)",
          line=dict(color="lightgray", width=1),
      )
  )
  fig.add_trace(
      go.Scatter(
          x=eval_var.index,
          y=eval_var,
          mode="lines",
          name="99% VaR Threshold ($)",
          line=dict(color="navy", width=1.5),
      )
  )

  breaches = eval_losses[exceptions == 1]
  fig.add_trace(
      go.Scatter(
          x=breaches.index,
          y=breaches,
          mode="markers",
          name="VaR Breaches",
          marker=dict(color="red", size=7, symbol="x"),
      )
  )
  fig.update_layout(
      title="Out-of-Sample Losses vs. 99% VaR Threshold",
      xaxis_title="Date",
      yaxis_title="Loss ($)",
      template="plotly_white",
  )
  st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: STRESS TESTING
# ---------------------------------------------------------
elif page == "5. Stress Testing":
  st.title("🔥 Macro & Crisis Stress Testing")

  port_rets, port_val, daily_pnl = simulate_portfolio_series(
      all_prices, np.repeat(1.0 / len(dow_tickers), len(dow_tickers)), 100000.0
  )
  curr_val = port_val.iloc[-1]

  scenarios = {
      "COVID Crash (2020)": (
          1 - (1 + port_rets.loc["2020-02-19":"2020-03-23"]).prod()
      )
      * curr_val,
      "Equity Down 20%": curr_val * 0.20,
      "Equity Down 10%": curr_val * 0.10,
      "Equity Down 5%": curr_val * 0.05,
      "Volatility Doubles": -(port_rets * 2.0).min() * curr_val,
      "Correlation Spikes to 1.0": -(port_rets * 1.5).min() * curr_val,
      "Sector Shock (-15%)": curr_val * 0.03,
  }

  df_sc = pd.DataFrame(
      [{"Scenario": k, "Stressed Loss ($)": v} for k, v in scenarios.items()]
  ).sort_values("Stressed Loss ($)", ascending=False)

  col1, col2 = st.columns([1, 2])
  with col1:
    st.subheader("Select Scenario")
    selected_sc = st.selectbox("Scenario", df_sc["Scenario"].tolist())
    loss_val = df_sc.loc[
        df_sc["Scenario"] == selected_sc, "Stressed Loss ($)"
    ].values[0]
    st.metric(f"Stressed Loss for {selected_sc}", f"${loss_val:,.2f}")
    st.metric("Portfolio Value After Shock", f"${curr_val - loss_val:,.2f}")

  with col2:
    fig = px.bar(
        df_sc,
        x="Stressed Loss ($)",
        y="Scenario",
        orientation="h",
        title="Stress Scenario Loss Comparison",
        color="Stressed Loss ($)",
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"), template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
