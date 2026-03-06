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
start_time = time.perf_counter()

warnings.filterwarnings("ignore")

# =====================================================
# 讀資料
# =====================================================
ticker = "QQQ"

df = pd.read_csv("QQQ.csv", parse_dates=['Date'])
df = df.sort_values('Date')
df = df[(df['Date'] >= '2010-01-01') & (df['Date'] <= '2024-12-31')] 
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')


df['LogReturn_10D'] = np.log(df['Close'] / df['Close'].shift(10)) # 10-days log return
df = df.dropna()
log_returns_10D = df['LogReturn_10D'].values


df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
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

res_gbm10 = minimize(
    gbm_neg_loglik,
    x0=[np.mean(log_returns_10D), np.std(log_returns_10D)],
    args=(log_returns_10D,),
    bounds=[(-1, 1), (1e-6, 1)],
    method="L-BFGS-B"
)

mu_gbm10, sigma_gbm10 = res_gbm10.x


print("\n===== GBM (Normal) MLE estimates =====")
print(f"mu     = {mu_gbm:.6f}")
print(f"sigma  = {sigma_gbm:.6f}")
print(f"Log-likelihood = {-res_gbm.fun:.2f}")


print("\n===== GBM (Normal) 10 days MLE estimates =====")
print(f"mu     = {mu_gbm10:.6f}")
print(f"sigma  = {sigma_gbm10:.6f}")
print(f"Log-likelihood = {-res_gbm10.fun:.2f}")



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
    theta, sigma, nu = params
    if sigma <= 0 or nu <= 0:
        return 1e10

    g = g_nodes[:, None]          # (GL, 1)
    w = g_weights[:, None]        # (GL, 1)

    x = data[None, :]             # (1, N)
    
    log_weight = (
        np.log(w) + (1/nu - 1) * np.log(g) - sp_gamma(1/nu)
    )


    log_pdf = norm.logpdf(
        x,
        loc=theta * g,
        scale=sigma * np.sqrt(g)
    )

    log_mix = logsumexp(log_weight + log_pdf, axis=0)

    return -np.sum(log_mix)



# =====================================================
# MLE estimation
# =====================================================

#init_params = np.array([0.002, 0.025, 0.3])
init_params = np.array([0, 0.02, 0.2])

bounds = [(-0.1, 0.1), (1e-4, 1), (1e-4, 1.5)]
# bounds = [(-0.5, 0.5), (1e-4, 1), (1e-4, 1.5)] # theta, sigma, nu

res = minimize(
    vg_neg_loglik_mixture_fast,
    init_params,
    args=(log_returns,),
    method="L-BFGS-B",
    bounds=bounds
)
theta_hat, sigma_hat, nu_hat = res.x

res10 = minimize(
    vg_neg_loglik_mixture_fast,
    init_params,
    args=(log_returns_10D,),
    method="L-BFGS-B",
    bounds=bounds
)
theta_hat10, sigma_hat10, nu_hat10 = res10.x

print("\n===== VG MLE estimates =====")
print(f"theta = {theta_hat:.6f}")
print(f"sigma = {sigma_hat:.6f}")
print(f"nu    = {nu_hat:.6f}")
print(f"Log-likelihood = {-res.fun:.2f}")

print("\n===== VG 10-days MLE estimates =====")
print(f"theta = {theta_hat10:.6f}")
print(f"sigma = {sigma_hat10:.6f}")
print(f"nu    = {nu_hat10:.6f}")
print(f"Log-likelihood = {-res10.fun:.2f}")




# =====================================================
# COS method: CDF / PDF
# =====================================================

def cos_cdf(x, theta, sigma, nu, N=1024, L=10):
    c1 = theta
    c2 = sigma**2 + nu*theta**2
    c4 = 3*nu*(sigma**4 + 2*sigma**2*theta**2 + theta**4)

    a = c1 - L*np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L*np.sqrt(c2 + np.sqrt(c4))

    k = np.arange(N)
    u = k*np.pi/(b-a)
    phi = vg_cf(u, theta, sigma, nu)

    Ak = (2/(b-a))*np.real(phi*np.exp(-1j*u*a))
    Ak[0] *= 0.5

    return np.sum(Ak*np.cos(u*(x-a)))


def cos_pdf(x, theta, sigma, nu, N=1024, L=10):
    c1 = theta
    c2 = sigma**2 + nu*theta**2
    c4 = 3*nu*(sigma**4 + 2*sigma**2*theta**2 + theta**4)

    a = c1 - L*np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L*np.sqrt(c2 + np.sqrt(c4))

    k = np.arange(N)
    u = k * np.pi / (b - a)

    phi = vg_cf(u, theta, sigma, nu)

    Ak = (2 / (b - a)) * np.real(phi * np.exp(-1j * u * a))
    Ak[0] *= 0.5

    return np.sum(Ak * np.cos(u * (x - a)))




# =====================================================
# Normal-Gamma mixture: CDF / VaR / CVaR
# =====================================================

def vg_pdf_mixture(x, theta, sigma, nu):
    g = g_nodes
    w = g_weights

    weight = w * g**(1/nu - 1) / sp_gamma(1/nu)
    pdf = norm.pdf(x, loc=theta*g, scale=sigma*np.sqrt(g))

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
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if vg_cdf_mixture(mid, theta, sigma, nu) < alpha:  # mid代表 
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



# =====================================================
# VaR / CVaR
# =====================================================
alpha = 0.01

VaR_vg = vg_var(alpha, theta_hat, sigma_hat, nu_hat)
CVaR_vg = vg_cvar(alpha, theta_hat, sigma_hat, nu_hat, VaR_vg)
VaR_vg10 = vg_var(alpha, theta_hat10, sigma_hat10, nu_hat10)
CVaR_vg10 = vg_cvar(alpha, theta_hat10, sigma_hat10, nu_hat10, VaR_vg10)

var_gbm = norm.ppf(alpha, loc=mu_gbm, scale=sigma_gbm)
cvar_gbm = mu_gbm - sigma_gbm * norm.pdf(norm.ppf(alpha)) / alpha
var_gbm10 = norm.ppf(alpha, loc=mu_gbm10, scale=sigma_gbm10)
cvar_gbm10 = mu_gbm10 - sigma_gbm10 * norm.pdf(norm.ppf(alpha)) / alpha

empirical_var = np.percentile(log_returns, alpha * 100)
empirical_var10 = np.percentile(log_returns_10D, alpha * 100)

print("\n===== Risk Measures =====")
print(f"VG {alpha*100:.1f}% VaR = {VaR_vg:.5f}")
print(f"VG {alpha*100:.1f}% CVaR = {CVaR_vg:.5f}")
print(f"GBM {alpha*100:.1f}% VaR = {var_gbm:.5f}")
print(f"GBM {alpha*100:.1f}% CVaR = {cvar_gbm:.5f}")
print(f"VG 10-days {alpha*100:.1f}% VaR = {VaR_vg10:.5f}")
print(f"VG 10-days {alpha*100:.1f}% CVaR = {CVaR_vg10:.5f}")
print(f"GBM 10-days {alpha*100:.1f}% VaR = {var_gbm10:.5f}")
print(f"GBM 10-days {alpha*100:.1f}% CVaR = {cvar_gbm10:.5f}")
print(f"Empirical {alpha*100:.1f}% quantile (VaR) = {empirical_var:.5f}")
print(f"Empirical 10-days {alpha*100:.1f}% quantile (VaR) = {empirical_var10:.5f}")



# =====================================================
# Empirical vs VG distribution plot
# =====================================================
x_grid = np.linspace(
    np.percentile(log_returns, 0.01),
    np.percentile(log_returns, 99.99),
    400 # points between 1 and 99 percentiles
)

# numerical pdf from COS-CDF
dx = x_grid[1] - x_grid[0]

vg_pdf = np.array([
    cos_pdf(x, theta_hat, sigma_hat, nu_hat)
    for x in x_grid
])

vg_pdf10 = np.array([
    cos_pdf(x, theta_hat10, sigma_hat10, nu_hat10)
    for x in x_grid
])

vg_pdf_scaling10 = np.array([
    cos_pdf(x, theta_hat * 10, sigma_hat * np.sqrt(10), nu_hat / (10 ** 0.5))
    for x in x_grid
])

gbm_pdf = norm.pdf(x_grid, loc=mu_gbm, scale=sigma_gbm)
gbm_pdf10 = norm.pdf(x_grid, loc=mu_gbm10, scale=sigma_gbm10)

plt.figure(figsize=(11, 6)) #代表圖的寬和高

plt.hist(
    log_returns_10D,
    bins=200, # bins是直方圖的柱子數量
    density=True,
    alpha=0.5, # alpha是透明度，代表顏色的深淺
    label="Empirical returns (10-days)"
)

plt.hist(
    log_returns,
    bins=200, # bins是直方圖的柱子數量
    density=True,
    alpha=0.5, # alpha是透明度，代表顏色的深淺
    label="Empirical returns (1-days)"
)

plt.plot(
    x_grid,
    vg_pdf,
    lw=1.6,  # lw是線條的寬度
    label="Estimated VG density"
)

plt.plot(
    x_grid,
    vg_pdf10,
    lw=1.6,  # lw是線條的寬度
    label="Estimated 10-days VG density"
)

'''
plt.plot(
    x_grid,
    vg_pdf_scaling10,
    lw=1.5,  # lw是線條的寬度
    color="purple",
    label="Estimated 10-days VG density (Scaling)"
)
'''

plt.plot(
    x_grid,
    gbm_pdf,
    lw=1.6,
    linestyle="--",
    label="Estimated GBM density"
)

plt.plot(
    x_grid,
    gbm_pdf10,
    lw=1.6,
    linestyle="--",
    label="Estimated 10-days GBM density"
)

plt.title(f"{ticker} log-return distributions \n Empirical vs Variance Gamma and GBM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 計算並顯示總運行時間
elapsed = time.perf_counter() - start_time
h, rem = divmod(elapsed, 3600)
m, s = divmod(rem, 60)
print(f"\n⏱️ Total runtime: {int(h)}h {int(m)}m {s:.2f}s")