import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 讀資料
ticker = "NASDAQ"
df = pd.read_csv("NASDAQ.csv", parse_dates=["Date"])
df = df.sort_values("Date")
df = df[(df['Date'] >= '2010-01-01') & (df['Date'] <= '2025-05-20')] 
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()


# 參數
window = 500
forecast_horizon = 10
alpha = 0.01
lambda_ewma = 0.97   # 加權參數

# ------------------------------
# Backtest window (固定區間)
# ------------------------------
backtest_start = pd.to_datetime("2023-01-01")
backtest_end   = pd.to_datetime("2024-12-31")



log_returns = df["LogReturn"].values

# ==============================
# Quantile / Tick Loss
# ==============================
def quantile_loss(y, q, alpha):
    return (alpha - (y < q)) * (y - q)

# ==============================
# Weighted Quantile
# ==============================
def weighted_quantile(values, weights, alpha):
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, alpha)]

# ==============================
# 儲存結果
# ==============================
var_list, cvar_list = [], []
actual_returns = []
breach_flags = []
quantile_loss_list = []
estimation_dates = []
dates = df['Date']


# 找出 backtest 區間對應 index
backtest_mask = (df['Date'] >= backtest_start) & (df['Date'] <= backtest_end)
backtest_indices = np.where(backtest_mask)[0]

# 因為我們是 forecast_horizon 天後才評估
# 所以要扣掉 forecast_horizon
valid_indices = backtest_indices[backtest_indices >= window]
valid_indices = valid_indices[valid_indices + forecast_horizon < len(df)]


# ==============================
# Rolling WHS
# ==============================
for i in valid_indices:

    # --- 歷史 log-return ---
    train_data = log_returns[i - window : i]

    # --- 歷史 10-day return ---
    hist_10d = np.array([
        np.sum(train_data[j : j + forecast_horizon])
        for j in range(len(train_data) - forecast_horizon + 1)
    ])

    N = len(hist_10d)

    # --- EWMA 權重（舊 → 新） ---
    raw_weights = (1 - lambda_ewma) * lambda_ewma ** np.arange(N-1, -1, -1) # arange(N-1, -1, -1)的意思是從 N-1 到 0 的數列
    weights = raw_weights / raw_weights.sum()

    # --- Weighted VaR / CVaR ---
    var_10_log = weighted_quantile(hist_10d, weights, alpha)

    tail_mask = hist_10d <= var_10_log
    cvar_10_log = np.sum(weights[tail_mask] * hist_10d[tail_mask]) / weights[tail_mask].sum()

    # --- simple return ---
    var_10 = np.exp(var_10_log) - 1
    cvar_10 = np.exp(cvar_10_log) - 1

    # --- 真實未來 ---
    future_log_return = np.sum(log_returns[i : i + forecast_horizon])
    future_return = np.exp(future_log_return) - 1

    ql = quantile_loss(future_return, var_10, alpha)

    var_list.append(var_10)
    cvar_list.append(cvar_10)
    actual_returns.append(future_return)
    breach_flags.append(future_return < var_10)
    quantile_loss_list.append(ql)
    estimation_dates.append(df["Date"].iloc[i + forecast_horizon])

# ------------------------------
# 畫圖（simple return）
# ------------------------------
# 畫圖
plt.figure(figsize=(14, 7))
plt.plot(estimation_dates, actual_returns, label=f'Actual {forecast_horizon}-day Return', color='orange')
plt.plot(estimation_dates, var_list, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% VaR (WHS)', color='green', linestyle='--')
plt.plot(estimation_dates, cvar_list, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% CVaR (WHS)', color='blue')
plt.fill_between(estimation_dates, var_list, actual_returns, where=np.array(breach_flags), 
                 color='red', alpha=0.3, label='VaR Breach')
plt.title(f'{ticker} WHS-based {forecast_horizon}-day {float((1-alpha)*100)}% VaR and CVaR Backtesting ({window}-day Rolling Estimation)')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------
# 統計分析
# ------------------------------
breach_array = np.array(breach_flags)
actual_array = np.array(actual_returns)
cvar_array = np.array(cvar_list)
quantile_loss_array = np.array(quantile_loss_list)

breach_losses = actual_array[breach_array]
avg_loss = np.mean(breach_losses)
avg_cvar = np.mean(cvar_array[breach_array])
violation_rate = np.mean(breach_array)
total_tests = len(breach_array)
num_breaches = np.sum(breach_array)

mean_ql = quantile_loss_array.mean()
mean_ql_breach = quantile_loss_array[breach_array].mean()

# ------------------------------
# 1. Kupiec Unconditional Coverage Test (UC)
# ------------------------------
kupiec_test = binomtest(num_breaches, total_tests, alpha, alternative='two-sided')

# ------------------------------
# Christoffersen IND and CC (robust, using log-likelihoods)
# ------------------------------
n00 = n01 = n10 = n11 = 0
for i in range(1, len(breach_array)):
    prev = int(breach_array[i - 1])
    curr = int(breach_array[i])
    if prev == 0 and curr == 0: n00 += 1
    elif prev == 0 and curr == 1: n01 += 1
    elif prev == 1 and curr == 0: n10 += 1
    elif prev == 1 and curr == 1: n11 += 1

# print counts for debugging
print(f"\nTransition counts: n00={n00}, n01={n01}, n10={n10}, n11={n11}")

total_trans = n00 + n01 + n10 + n11
# basic checks: need at least one transition to compute IND, and at least 2 obs
if total_tests < 2 or total_trans == 0:
    ind_pval = np.nan
    cc_pval = np.nan
    print("⚠️ Too few observations or no transitions — IND/CC tests not applicable.")
else:
    eps = 1e-12
    # empirical conditional probs (clip to avoid log(0) issues)
    pi_0 = (n01) / (n00 + n01) if (n00 + n01) > 0 else np.nan
    pi_1 = (n11) / (n10 + n11) if (n10 + n11) > 0 else np.nan
    pi_all = (n01 + n11) / total_trans

    pi_0_c = np.clip(pi_0 if not np.isnan(pi_0) else pi_all, eps, 1-eps)
    pi_1_c = np.clip(pi_1 if not np.isnan(pi_1) else pi_all, eps, 1-eps)
    pi_all_c = np.clip(pi_all, eps, 1-eps)

    # log-likelihoods (safer than multiplying probabilities)
    logL_uncond = (n00 + n10) * np.log(1 - pi_all_c) + (n01 + n11) * np.log(pi_all_c)
    logL_cond = (n00) * np.log(1 - pi_0_c) + (n01) * np.log(pi_0_c) + \
                (n10) * np.log(1 - pi_1_c) + (n11) * np.log(pi_1_c)

    lr_ind = -2 * (logL_uncond - logL_cond)
    # numerical safety: lr_ind >= 0 theoretically, but small negative may appear due to numerics
    if lr_ind < 0 and lr_ind > -1e-8:
        lr_ind = 0.0
    if lr_ind < 0:
        print(f"⚠️ lr_ind negative (numerical?): {lr_ind:.4e}")

    ind_pval = 1 - chi2.cdf(lr_ind, df=1)

    # CC: compare null that only matches proportion (Kupiec) vs alt (cond)
    # log-likelihood under null (proportion only)
    pi_null = np.clip(num_breaches / total_tests, eps, 1-eps)
    logL_null = (total_tests - num_breaches) * np.log(1 - pi_null) + num_breaches * np.log(pi_null)
    logL_alt = logL_cond  # alternative uses conditional-likelihood

    lr_cc = -2 * (logL_null - logL_alt)
    if lr_cc < 0 and lr_cc > -1e-8:
        lr_cc = 0.0
    cc_pval = 1 - chi2.cdf(lr_cc, df=2)

    # diagnostic print
    print(f"pi_0 = {pi_0}, pi_1 = {pi_1}, pi_all = {pi_all}")
    print(f"logL_uncond = {logL_uncond:.6f}, logL_cond = {logL_cond:.6f}")
    print(f"lr_ind = {lr_ind:.6f}, IND p-value = {ind_pval:.6f}")
    print(f"lr_cc  = {lr_cc:.6f}, CC  p-value = {cc_pval:.6f}")

# ------------------------------
# 輸出結果
# ------------------------------

print(f'違約時平均實際損失   : {avg_loss:.4f}')
print(f'CVaR 預估值平均       : {avg_cvar:.4f}')
print(f'總違約次數             : {num_breaches}（佔 {violation_rate * 100:.2f}%）')

print(f'\nKupiec UC Test p-value   : {kupiec_test.pvalue:.4e}')
print(f'Christoffersen IND p-value: {ind_pval:.4e}')
print(f'Christoffersen CC  p-value: {cc_pval:.4e}')

if kupiec_test.pvalue < 0.05:
    print("❌ UC Test：違約比例與預期顯著不同")
else:
    print("✅ UC Test：違約比例合理")

if ind_pval < 0.05:
    print("❌ IND Test：違約不獨立（有 clustering）")
else:
    print("✅ IND Test：違約事件可能獨立")

if cc_pval < 0.05:
    print("❌ CC Test：整體模型不通過（比例 + 串聯都不好）")
else:
    print("✅ CC Test：模型通過條件覆蓋檢定")

print(f'\n平均 Quantile Loss (Tick Loss)        : {mean_ql:.6f}')
print(f'違約時平均 Quantile Loss              : {mean_ql_breach:.6f}')