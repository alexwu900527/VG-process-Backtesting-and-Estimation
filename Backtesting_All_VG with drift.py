import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import norm, binomtest, chi2
from scipy.optimize import minimize
from arch import arch_model
from numpy.polynomial.laguerre import laggauss
from scipy.special import logsumexp
from scipy.special import gamma as sp_gamma, gammaln as sp_gammaln
from scipy.integrate import quad
import warnings
import time
start_time = time.perf_counter()

warnings.filterwarnings("ignore")

# ==============================
# 基本設定
# ==============================

ticker = "NASDAQ"
alpha = 0.01
window_vg = 600
window_gbm = 750
window_whs = 200
window_garch = 750

forecast_horizon = 1

backtest_start = pd.to_datetime("2025-01-01")
backtest_end   = pd.to_datetime("2025-05-20")


# ==============================
# 讀資料
# ==============================

df = pd.read_csv(f"{ticker}.csv", parse_dates=["Date"])
df = df.sort_values("Date")
df = df[(df['Date'] >= '2010-01-01') & (df['Date'] <= '2025-05-20')]
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna()
log_returns = df["LogReturn"].values


# ==============================
# Quantile loss
# ==============================

def quantile_loss(y, q, alpha):
    return (alpha - (y < q)) * (y - q)


actual_returns = []
estimation_dates = []

# 儲存結果（全部是 simple return）
var_list_vg, cvar_list_vg = [], []
breach_flags_vg = []
quantile_loss_list_vg = []

var_list_gbm, cvar_list_gbm = [], []
breach_flags_gbm = []
quantile_loss_list_gbm = []

var_list_whs, cvar_list_whs = [], []
breach_flags_whs = []
quantile_loss_list_whs = []

var_list_garch, cvar_list_garch = [], []
breach_flags_garch = []
quantile_loss_list_garch = []


dates = df['Date']
log_returns = df['LogReturn'].values



# VG


def empirical_var(data, alpha):
    return np.quantile(data, alpha)

def empirical_cvar(data, alpha):
    var_alpha = np.quantile(data, alpha)
    return data[data <= var_alpha].mean()


# =====================================================
# VG characteristic function 
# =====================================================
def vg_cf(u, theta, sigma, nu):
    return (1 - 1j*theta*nu*u + 0.5*sigma**2*nu*u**2) ** (-1/nu)


# =====================================================
# VG likelihood via Normal–Gamma mixture (Gauss–Laguerre)
# =====================================================

GL_N = 40
g_nodes, g_weights = laggauss(GL_N)

def vg_neg_loglik_mixture_fast(params, data):
    mu, theta, sigma, nu = params

    if sigma <= 0 or nu <= 0:
        return 1e10

    g = g_nodes[:, None]
    w = g_weights[:, None]
    x = data[None, :]

    log_weight = (
        np.log(w) + (1/nu - 1) * np.log(g) - sp_gammaln(1/nu) #大的window用gamma 小的window用gammaln
    )

    log_pdf = norm.logpdf(
        x,
        loc=mu + theta * g,
        scale=sigma * np.sqrt(g)
    )

    log_mix = logsumexp(log_weight + log_pdf, axis=0)

    return -np.sum(log_mix)




# =====================================================
# COS method: CDF / VaR / CVaR
# =====================================================

def cos_cdf(x, mu,theta, sigma, nu, N=1024, L=10):
    c1 = mu + theta
    c2 = sigma**2 + nu*theta**2
    c4 = 3*nu*(sigma**4 + 2*sigma**2*theta**2 + theta**4)

    a = c1 - L*np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L*np.sqrt(c2 + np.sqrt(c4))

    k = np.arange(N)
    u = k*np.pi/(b-a)
    phi = vg_cf(u, mu, theta, sigma, nu)

    Ak = (2/(b-a))*np.real(phi*np.exp(-1j*u*a))
    Ak[0] *= 0.5

    return np.sum(Ak*np.cos(u*(x-a)))


def cos_pdf(x, mu,theta, sigma, nu, N=1024, L=10):
    c1 = mu + theta
    c2 = sigma**2 + nu*theta**2
    c4 = 3*nu*(sigma**4 + 2*sigma**2*theta**2 + theta**4)

    a = c1 - L*np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L*np.sqrt(c2 + np.sqrt(c4))

    k = np.arange(N)
    u = k * np.pi / (b - a)

    phi = vg_cf(u, mu, theta, sigma, nu)

    Ak = (2 / (b - a)) * np.real(phi * np.exp(-1j * u * a))
    Ak[0] *= 0.5

    return np.sum(Ak * np.cos(u * (x - a)))



def vg_pdf_mixture(x, mu, theta, sigma, nu):
    g = g_nodes
    w = g_weights

    weight = w * g**(1/nu - 1) / sp_gamma(1/nu)
    pdf = norm.pdf(x, loc=mu + theta*g, scale=sigma*np.sqrt(g))

    return np.sum(weight * pdf)


def vg_cdf_mixture(x, theta, sigma, nu):
    val, _ = quad(
        lambda t: vg_pdf_mixture(t, theta, sigma, nu),
        -np.inf,
        x,
        limit=200
    )
    return val

def vg_var(alpha, theta, sigma, nu):
    lo, hi = -2.0, 0.2   # 對 10-days 很安全
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if vg_cdf_mixture(mid, theta, sigma, nu) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def vg_cvar(alpha, theta, sigma, nu, var_alpha):
    num, _ = quad(
        lambda x: x * vg_pdf_mixture(x, theta, sigma, nu),
        -np.inf,
        var_alpha,
        limit=200
    )
    return num / alpha

# 找出 backtest 區間對應 index
backtest_mask = (df['Date'] >= backtest_start) & (df['Date'] <= backtest_end)
backtest_indices = np.where(backtest_mask)[0]


# 因為我們是 forecast_horizon 天後才評估
# 所以要扣掉 forecast_horizon
valid_indices = backtest_indices[backtest_indices >= window_vg]
valid_indices = valid_indices[valid_indices + forecast_horizon < len(df)]

# ------------------------------
# Rolling estimation
# ------------------------------
for i in valid_indices:
    train_data = log_returns[i - window_vg:i]
    
    GL_N = 40
    g_nodes, g_weights = laggauss(GL_N)

    # MLE estimation
    init_params = np.array([0.0, 0.02, 0.2])

    res = minimize(
        vg_neg_loglik_mixture_fast,
        init_params,
        args=(train_data,),
        bounds = [(-0.01, 0.01), (1e-4, 1), (1e-4, 1.5)], # theta, sigma, nu
        method="L-BFGS-B",
    )

    theta_hat, sigma_hat, nu_hat = res.x


    # === 10-day log-return distribution ===

    # ===============================
    # 1️⃣ VG 1-day VaR / CVaR
    # ===============================
    var_1_log = vg_var(alpha, theta_hat, sigma_hat, nu_hat)
    cvar_1_log = vg_cvar(alpha, theta_hat, sigma_hat, nu_hat, var_1_log)

    var_1 = np.exp(var_1_log) - 1
    cvar_1 = np.exp(cvar_1_log) - 1


    # ===============================
    # 2️⃣ Empirical 1-day / 10-day
    # ===============================

    # 1-day simple return (training window)
    train_simple = np.exp(train_data) - 1 #所有的window用simple return

    # 10-day overlapping log-return
    train_10_log = np.array([
        np.sum(train_data[j:j+forecast_horizon])
        for j in range(len(train_data)-forecast_horizon+1)
    ])

    train_10_simple = np.exp(train_10_log) - 1

    emp_var_1 = empirical_var(train_simple, alpha)
    emp_cvar_1 = empirical_cvar(train_simple, alpha)

    emp_var_10 = empirical_var(train_10_simple, alpha)
    emp_cvar_10 = empirical_cvar(train_10_simple, alpha)


    # ===============================
    # 3️⃣ Multiplier
    # ===============================

    multiplier_var = emp_var_10 / emp_var_1
    multiplier_cvar = emp_cvar_10 / emp_cvar_1


    # ===============================
    # 4️⃣ Scaled VG 10-day
    # ===============================

    var_10 = multiplier_var * var_1
    cvar_10 = multiplier_cvar * cvar_1



    future_log_return = np.sum(log_returns[i:i + forecast_horizon])
    future_return = np.exp(future_log_return) - 1

    ql = quantile_loss(future_return, var_10, alpha)

    var_list_vg.append(var_10)
    cvar_list_vg.append(cvar_10)
    actual_returns.append(future_return)
    breach_flags_vg.append(future_return < var_10)
    quantile_loss_list_vg.append(ql)
    estimation_dates.append(df['Date'].iloc[i + forecast_horizon])






# GBM


# ------------------------------
# GBM log-return MLE
# ------------------------------
def gbm_neg_log_likelihood(params, data, dt=1):
    mu, sigma = params
    if sigma <= 0:
        return 1e10
    mu_adj = (mu - 0.5 * sigma**2) * dt
    sigma_adj = sigma * np.sqrt(dt)
    return -np.sum(norm.logpdf(data, loc=mu_adj, scale=sigma_adj))

valid_indices = backtest_indices[backtest_indices >= window_gbm]
valid_indices = valid_indices[valid_indices + forecast_horizon < len(df)]


# ------------------------------
# Rolling estimation
# ------------------------------
# ------------------------------
# Rolling estimation (固定 backtest 區間)
# ------------------------------

for i in valid_indices:

    train_data = log_returns[i - window_gbm:i]

    result = minimize(
        gbm_neg_log_likelihood,
        x0=[0, 0.1],
        args=(train_data,),
        bounds=[(-1, 1), (1e-6, 1)],
        method="L-BFGS-B"
    )

    mu_hat, sigma_hat = result.x

    # === 10-day log-return distribution ===
    mu_10 = (mu_hat - 0.5 * sigma_hat**2) * forecast_horizon
    sigma_10 = sigma_hat * np.sqrt(forecast_horizon)

    var_10_log = norm.ppf(alpha, loc=mu_10, scale=sigma_10)
    cvar_10_log = mu_10 - sigma_10 * norm.pdf(norm.ppf(alpha)) / alpha

    # === 轉成 simple return ===
    var_10 = np.exp(var_10_log) - 1
    cvar_10 = np.exp(cvar_10_log) - 1

    future_log_return = np.sum(log_returns[i:i + forecast_horizon])
    future_return = np.exp(future_log_return) - 1

    ql = quantile_loss(future_return, var_10, alpha)

    var_list_gbm.append(var_10)
    cvar_list_gbm.append(cvar_10)
    breach_flags_gbm.append(future_return < var_10)
    quantile_loss_list_gbm.append(ql)




# WHS

# ==============================
# Weighted Quantile
# ==============================
def weighted_quantile(values, weights, alpha):
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cw = np.cumsum(w)
    return v[np.searchsorted(cw, alpha)]


lambda_ewma = 0.97   # 加權參數


# 因為我們是 forecast_horizon 天後才評估
# 所以要扣掉 forecast_horizon
valid_indices = backtest_indices[backtest_indices >= window_whs]
valid_indices = valid_indices[valid_indices + forecast_horizon < len(df)]


# ==============================
# Rolling WHS
# ==============================
for i in valid_indices:

    # --- 歷史 log-return ---
    train_data = log_returns[i - window_whs : i]

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

    var_list_whs.append(var_10)
    cvar_list_whs.append(cvar_10)
    breach_flags_whs.append(future_return < var_10)
    quantile_loss_list_whs.append(ql)



# GARCH


# 因為我們是 forecast_horizon 天後才評估
# 所以要扣掉 forecast_horizon
valid_indices = backtest_indices[backtest_indices >= window_garch]
valid_indices = valid_indices[valid_indices + forecast_horizon < len(df)]

# -------------------------------
# 🔥 Rolling GARCH(1,1) estimation and forecast
# -------------------------------
for i in valid_indices:

    train_data = log_returns[i-window_garch:i]

    # rescale
    train_scaled = 100 * train_data

    model = arch_model(
        train_scaled,
        vol='Garch',
        p=1, q=1,
        mean='Constant',
        dist='normal'
    )

    try:
        res = model.fit(disp='off', options={'maxiter': 200})
    except:
        print(f"⚠️ WARNING: GARCH failed at window ending {i}, skipping.")
        continue

    # ===============================
    # ✅ 1. Stationarity check
    # ===============================
    alpha_hat = res.params['alpha[1]']
    beta_hat  = res.params['beta[1]']


    #if alpha_hat + beta_hat >= 0.999:
        # Near-IGARCH or non-stationary → 理論上不應做 multi-step VaR
    #    continue

    # ===============================
    # ✅ 2. Mean & variance forecast (arch built-in)
    # ===============================
    mu_hat = res.params['mu'] / 100

    # arch 的 forecast（variance）
    forecasts = res.forecast(horizon=forecast_horizon)

    # 取最後一期、未來 1~10 期的 conditional variance
    h_forecast = forecasts.variance.values[-1] / (100**2)

    # ===============================
    # ✅ 3. 10-day aggregation
    # ===============================
    sigma_10 = np.sqrt(h_forecast.sum())
    mu_10 = mu_hat * forecast_horizon

    var_10_log = norm.ppf(alpha, mu_10, sigma_10)
    cvar_10_log = mu_10 - sigma_10 * norm.pdf(norm.ppf(alpha)) / alpha

    # === 轉成 simple return ===
    var_10 = np.exp(var_10_log) - 1
    cvar_10 = np.exp(cvar_10_log) - 1

    future_log_return = np.sum(log_returns[i:i+forecast_horizon])
    future_return = np.exp(future_log_return) - 1

    ql = quantile_loss(future_return, var_10, alpha)
    quantile_loss_list_garch.append(ql)

    var_list_garch.append(var_10)
    cvar_list_garch.append(cvar_10)
    breach_flags_garch.append(future_return < var_10)




# ------------------------------
# 畫圖（simple return）
# ------------------------------
# 畫圖
plt.figure(figsize=(14, 7))
plt.plot(estimation_dates, actual_returns, label=f'Actual {forecast_horizon}-day Return', color='black', lw=2.5)
plt.plot(estimation_dates, var_list_vg, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% VaR (VG)', color='green', lw=2)
plt.plot(estimation_dates, cvar_list_vg, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% CVaR (VG)', color='green', linestyle='--', lw=1.5)
plt.plot(estimation_dates, var_list_garch, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% VaR (GARCH)', color='blue', lw=2)
plt.plot(estimation_dates, cvar_list_garch, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% CVaR (GARCH)', color='blue', linestyle='--', lw=1.5)
plt.plot(estimation_dates, var_list_gbm, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% VaR (GBM)', color='orange', lw=2)
plt.plot(estimation_dates, cvar_list_gbm, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% CVaR (GBM)', color='orange', linestyle='--', lw=1.5)
plt.plot(estimation_dates, var_list_whs, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% VaR (WHS)', color='red', lw=2)
plt.plot(estimation_dates, cvar_list_whs, label=f'{forecast_horizon}-day {float((1-alpha)*100)}% CVaR (WHS)', color='red', linestyle='--', lw=1.5)
plt.title(f'{ticker} {forecast_horizon}-day {float((1-alpha)*100)}% VaR and CVaR Backtesting')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# ==============================
# 統計結果
# ==============================

models = {
    "VG": (breach_flags_vg, cvar_list_vg, quantile_loss_list_vg),
    "GBM": (breach_flags_gbm, cvar_list_gbm, quantile_loss_list_gbm),
    "WHS": (breach_flags_whs, cvar_list_whs, quantile_loss_list_whs),
    "GARCH": (breach_flags_garch, cvar_list_garch, quantile_loss_list_garch)
}


for model_name, (breach_flags, cvar_list, quantile_loss_list) in models.items():

    actual_array = np.array(actual_returns)
    breach_array = np.array(breach_flags)
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
    # Kupiec LR POF (Likelihood Ratio)
    # ------------------------------
    pi_hat = num_breaches / total_tests

    eps = 1e-12
    pi_hat = np.clip(pi_hat, eps, 1-eps)
    alpha_c = np.clip(alpha, eps, 1-eps)

    logL_null = (total_tests - num_breaches) * np.log(1 - alpha_c) + num_breaches * np.log(alpha_c)
    logL_alt  = (total_tests - num_breaches) * np.log(1 - pi_hat) + num_breaches * np.log(pi_hat)

    lr_pof = -2 * (logL_null - logL_alt)

    if lr_pof < 0 and lr_pof > -1e-8:
        lr_pof = 0.0

    pval_pof = 1 - chi2.cdf(lr_pof, df=1)

    print(f'\nKupiec LR POF statistic : {lr_pof:.6f}')
    print(f'Kupiec LR POF p-value   : {pval_pof:.4e}')

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


    print(f'\n平均 Quantile Loss (Tick Loss)        : {mean_ql:.6f}')
    print(f'違約時平均 Quantile Loss              : {mean_ql_breach:.6f}')