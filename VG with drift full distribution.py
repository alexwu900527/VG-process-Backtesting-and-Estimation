import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import gamma as sp_gamma, gammaln as sp_gammaln
from numpy.polynomial.laguerre import laggauss
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.integrate import quad
import warnings
import time
from scipy.stats import skew, kurtosis

start_time = time.perf_counter()

warnings.filterwarnings("ignore")

# =====================================================
# 讀資料
# =====================================================
ticker = "S&P500"  # 可替換成其他股票指數或資產
alpha = 0.01
d = 1

df = pd.read_csv(f"{ticker}.csv", parse_dates=['Date'])
df = df.sort_values('Date')
df = df[(df['Date'] >= '2021-01-01') & (df['Date'] <= '2024-12-31')] 
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')


df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(d))
df = df.dropna()
log_returns = df['LogReturn'].values


# =====================================================
# GBM (Normal) MLE for log-returns
# =====================================================
def gbm_neg_loglik(params, data, dt=1):
    mu, sigma = params
    if sigma <= 0:
        return 1e10
    mu_adj = (mu - 0.5 * sigma**2) * dt
    sigma_adj = sigma * np.sqrt(dt)
    return -np.sum(norm.logpdf(data, loc=mu_adj, scale=sigma_adj))


res_gbm = minimize(
    gbm_neg_loglik,
    x0=[np.mean(log_returns), np.std(log_returns)],
    args=(log_returns,),
    bounds=[(-1, 1), (1e-6, 1)],
    method="L-BFGS-B"
)

mu_gbm, sigma_gbm = res_gbm.x


print("\n===== GBM (Normal) MLE estimates =====")
print(f"mu     = {mu_gbm:.6f}")
print(f"sigma  = {sigma_gbm:.6f}")
print(f"Log-likelihood = {-res_gbm.fun:.2f}")


# =====================================================
# VG characteristic function
# =====================================================
def vg_cf(u, mu, theta, sigma, nu):
    return np.exp(1j * u * mu) * \
           (1 - 1j*theta*nu*u + 0.5*sigma**2*nu*u**2) ** (-1/nu)


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
        np.log(w) + (1/nu - 1) * np.log(g) - sp_gammaln(1/nu) #都用gammaln型態，避免數值不穩定
    )

    log_pdf = norm.logpdf(
        x,
        loc=mu + theta * g,
        scale=sigma * np.sqrt(g)
    )

    log_mix = logsumexp(log_weight + log_pdf, axis=0)

    return -np.sum(log_mix)

# =====================================================
# MLE estimation
# =====================================================
#init_params = np.array([0.002, 0.025, 0.3])
# bounds = [(-0.5, 0.5), (1e-4, 1), (1e-4, 1.5)] # theta, sigma, nu

init_params = np.array([0.0, 0.0, 0.02, 0.2])
bounds = [
    (-0.1, 0.1),   # mu
    (-0.1, 0.1),   # theta
    (1e-4, 1),     # sigma
    (1e-4, 1.5)    # nu
]

res = minimize(
    vg_neg_loglik_mixture_fast,
    init_params,
    args=(log_returns,),
    method="L-BFGS-B",
    bounds=bounds
)

mu_hat, theta_hat, sigma_hat, nu_hat = res.x



print("===== VG MLE estimates (in-sample) =====")
print(f"mu    = {mu_hat:.6f}")
print(f"theta = {theta_hat:.6f}")
print(f"sigma = {sigma_hat:.6f}")
print(f"nu    = {nu_hat:.6f}")
print(f"Log-likelihood = {-res.fun:.2f}")





# =====================================================
# COS method: CDF 
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


# =====================================================
# Normal-Gamma mixture: CDF / VaR / CVaR
# =====================================================


def vg_pdf_mixture(x,mu,theta, sigma, nu):
    g = g_nodes
    w = g_weights

    weight = w * g**(1/nu - 1) / sp_gamma(1/nu)
    pdf = norm.pdf(x, loc=mu + theta*g, scale=sigma*np.sqrt(g))

    return np.sum(weight * pdf)


def vg_cdf_mixture(x, mu, theta, sigma, nu):
    val, _ = quad(
        lambda t: vg_pdf_mixture(t, mu, theta, sigma, nu),
        -np.inf,
        x,
        limit=200
    )
    return val

def vg_var(alpha, mu, theta, sigma, nu):
    lo, hi = -2.0, 0.2   # 對 10-days 很安全
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if vg_cdf_mixture(mid, mu, theta, sigma, nu) < alpha:  # mid代表 
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def vg_cvar(alpha, mu, theta, sigma, nu, var_alpha):
    num, _ = quad(
        lambda x: x * vg_pdf_mixture(x, mu, theta, sigma, nu),
        -np.inf,
        var_alpha,
        limit=200
    )
    return num / alpha

# =====================================================
# VaR / CVaR
# =====================================================

VaR_vg = vg_var(alpha, mu_hat,theta_hat, sigma_hat, nu_hat)
CVaR_vg = vg_cvar(alpha, mu_hat, theta_hat, sigma_hat, nu_hat, VaR_vg)

vg_var_simple = np.exp(VaR_vg) - 1
vg_cvar_simple = np.exp(CVaR_vg) - 1


var_gbm = norm.ppf(alpha, loc=mu_gbm, scale=sigma_gbm)
cvar_gbm = mu_gbm - sigma_gbm * norm.pdf(norm.ppf(alpha)) / alpha

gbm_var_simple = np.exp(var_gbm) - 1
gbm_cvar_simple = np.exp(cvar_gbm) - 1

empirical_var = np.percentile(log_returns, alpha * 100)
empirical_var_simple = np.exp(empirical_var) - 1

tail_losses = log_returns[log_returns <= empirical_var]
empirical_cvar = tail_losses.mean()
empirical_cvar_simple = np.exp(empirical_cvar) - 1


print("\n===== Risk Measures =====")
print(f"VG {alpha*100:.1f}% Simple VaR = {vg_var_simple:.5f}")
print(f"VG {alpha*100:.1f}% Simple CVaR = {vg_cvar_simple:.5f}")
print(f"GBM {alpha*100:.1f}% Simple VaR = {gbm_var_simple:.5f}")
print(f"GBM {alpha*100:.1f}% Simple CVaR = {gbm_cvar_simple:.5f}")
print(f"Empirical {alpha*100:.1f}% quantile (VaR) = {empirical_var_simple:.5f}")
print(f"Empirical {alpha*100:.1f}% quantile (CVaR) = {empirical_cvar_simple:.5f}")


# =====================================================
# Skewness & Kurtosis
# =====================================================

# Empirical
skew_emp = skew(log_returns)
kurt_emp = kurtosis(log_returns, fisher=False)  # fisher=False 代表不減去3，直接給出常態分布的kurtosis值（3）作為基準

# GBM (Normal)
skew_gbm = 0
kurt_gbm = 3

# VG theoretical moments
# Var = sigma^2 + nu * theta^2
var_vg = sigma_hat**2 + nu_hat * theta_hat**2

# Skewness
skew_vg = (2 * theta_hat**3 * nu_hat**2 + 3 * sigma_hat**2 * theta_hat * nu_hat) / (var_vg ** (3/2))

# Kurtosis
kurt_vg = 3 + (
    (   3 * sigma_hat**4 * nu_hat + 12 * sigma_hat**2 * theta_hat**2 * nu_hat**2 + 6 * theta_hat**4 * nu_hat**3
        + 3 * sigma_hat**4 + 6 * sigma_hat**2 * theta_hat**2 * nu_hat + 3 * theta_hat**4 * nu_hat**2
    )
    / (var_vg ** 2)
)

print("\n===== Skewness & Kurtosis =====")
print(f"Empirical Skewness = {skew_emp:.4f}")
print(f"Empirical Kurtosis = {kurt_emp:.4f}")

print(f"GBM Skewness = {skew_gbm:.4f}")
print(f"GBM Kurtosis = {kurt_gbm:.4f}")

print(f"VG Skewness = {skew_vg:.4f}")
print(f"VG Kurtosis = {kurt_vg:.4f}")

# =====================================================
# Empirical vs VG distribution plot
# =====================================================
x_grid = np.linspace(
    np.percentile(log_returns, 0.1),
    np.percentile(log_returns, 99.9),
    400 # points between 1 and 99 percentiles
)

# numerical pdf from COS-CDF
dx = x_grid[1] - x_grid[0]

vg_pdf = np.array([
    cos_pdf(x, mu_hat, theta_hat, sigma_hat, nu_hat)
    for x in x_grid
])


gbm_pdf = norm.pdf(x_grid, loc=mu_gbm, scale=sigma_gbm)


plt.figure(figsize=(11, 6)) #代表圖的寬和高

plt.hist(
    log_returns,
    bins=200, # bins是直方圖的柱子數量
    density=True,
    alpha=0.4, # alpha是透明度，代表顏色的深淺
    label="Empirical returns"
)

plt.plot(
    x_grid,
    vg_pdf,
    lw=1.6,  # lw是線條的寬度
    label="Estimated VG with drift density"
)



plt.plot(
    x_grid,
    gbm_pdf,
    lw=1.6,
    linestyle="--",
    label="Estimated GBM density"
)

plt.title(f"{ticker} {d}-day log-return distribution\nEmpirical vs Variance Gamma and GBM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
# =====================================================
# Left-tail zoomed plot (GBM vs VG)
# =====================================================

alpha_tail = 0.3
alpha_extreme = 0.0001

x_left = np.percentile(log_returns, alpha_extreme * 100)
x_right = np.percentile(log_returns, alpha_tail * 100)

x_grid_left = np.linspace(x_left, x_right, 400)

vg_pdf_left = np.array([
    cos_pdf(x, mu_hat, theta_hat, sigma_hat, nu_hat)
    for x in x_grid_left
])

gbm_pdf_left = norm.pdf(
    x_grid_left,
    loc=mu_gbm,
    scale=sigma_gbm
)

plt.figure(figsize=(10, 6))

# empirical histogram (left tail only)
plt.hist(
    log_returns[log_returns <= x_right],
    bins=80,
    density=True,
    alpha=0.45, 
    label="Empirical (left tail)"
)

# VG
plt.plot(
    x_grid_left,
    vg_pdf_left,
    lw=2.0,
    label="VG density (left tail)"
)

# GBM
plt.plot(
    x_grid_left,
    gbm_pdf_left,
    lw=2.0,
    linestyle="--",
    label="GBM density (left tail)"
)

# VaR reference line（可選但很有說服力）
plt.axvline(
    VaR_vg,
    color="black",
    linestyle=":",
    lw=1.5,
    label="VG 1% VaR"
)

plt.axvline(
    var_gbm,
    color="gray",
    linestyle=":",
    lw=1.5,
    label="GBM 1% VaR"
)

plt.title(
    f"{ticker} 10-days log-return\nLeft-tail zoom: GBM vs VG"
)
plt.xlabel("Log-return")
plt.ylabel("Density")

#plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''

# 計算並顯示總運行時間
elapsed = time.perf_counter() - start_time
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"\n⏱️ Total runtime: {int(h)}h {int(m)}m {s:.2f}s")