# First I will Import the OSLO stock files
import numpy as np
import pandas as pd

# ======================= PART A) FIRST ===================================================
# =========================================================================================

# ============= Firstly we load the adjusted Prices sheet ==========================

# The workbook has two sheets: "ReadMe" and "Adjusted Prices".
# First row of "Adjusted Prices" is a title.
# Second row contains column headers (Date, EQNR, DNB, TEL, NAS, NHY).
# Skp the title row and use the header now.

prices = pd.read_excel(
    "oslo-stocks-2023-2025.xlsx",
    sheet_name="Adjusted Prices",
    header=2,            # row index 2 (0-indexed) holds the real header
)

# Clean any stray whitespace in the column names
prices.columns = prices.columns.astype(str).str.strip()

# Set Date as index AFTER reading (avoids the whitespace match problem)
prices = prices.set_index("Date")
prices.index = pd.to_datetime(prices.index)

# Drop fully-empty rows
prices = prices.dropna(how="all")

print("Shape:", prices.shape)                        # expect (751, 5)
print("Columns:", list(prices.columns))
print("Date range:", prices.index[0].date(), "to", prices.index[-1].date())
print(prices.head())

# ============= Now Compute Daily Simple returns =====================================

# The formula is r_{i,k} = (S_i(t_k) - S_i(t_{k-1})) / S_i(t_{k-1})
returns = prices.pct_change().dropna(how = "all")
print("\nReturns shape: ", returns.shape) # Expect (750,5) now

# ================= Now Sample moments (per stock) ===================================
# Sample mean:     mu_hat_i = (1/n) sum_k r_{i,k} = COM of distributions in R^N
# Sample variance: nu_hat_ii = (1/(n-1)) sum_k (r_{i,k} - mu_hat_i)^2
# Sample vol:      sigma_hat_i = sqrt(nu_hat_ii)

mu_hat = returns.mean()
nu_hat = returns.var(ddof=1) # ddof = 1 gives n-1 denominator 
sigma_hat = np.sqrt(nu_hat)

print("\nSample mean (daily): ", mu_hat)
print("\nSample variance (daily) ",nu_hat)
print("\nSample volatility (daily)", sigma_hat)

# ================== Sample Covariance ===============================================

# V_hat_ij = (1/(n-1)) sum_k (r_{i,k} - mu_hat_i)(r_{j,k} - mu_hat_j)
v_hat = returns.cov(ddof =1).values # numpy array, shape (5,5)

print("\nCovariance matrix V_hat: ")
print(pd.DataFrame(v_hat,index = returns.columns, columns = returns.columns ).round(8))

# ==================== Sanity Checks ==================================================

# ------------------ Check 1: Symmetry of V_hat -----------------------
assert np.allclose(v_hat,v_hat.T), "V_hat is not symmetric"
print("\nCheck 1-> V_hat symmetric: PASS")

# ----- Chekc 2: Diagonal elemens of V_hat equals nu_hat (the variance) ---------
assert np.allclose(np.diag(v_hat), nu_hat.values), "diag(V_hat) != nu_hat values"
print("Check 2 -> diag(V_hat) == nu_hat: PASS")

# ---------------- Check 3: Positive definite V_hat (eigen values > 0)
eigs = np.linalg.eigvalsh(v_hat)
assert (eigs > 0 ).all(), "V_hat not positive definite"
print(f"Check 3 -> V_hat positive definite: PASS (min eig = {eigs.min():.3e})")



# ----------------- Check 4: number of returns = number of prices -1 ------------
assert len(returns) == len(prices) - 1
print(f"Check 4-> n_returns = n_prices -1 = {len(returns)}: PASS")

# ======================= PART B) Part 2 ===================================================
# Replace nu->nu_hat and  V->V_hat, then compute A_hat,B_hat,C_hat and empirical global
# min-var portfolio. Report its weigths, expected return and risk. Check budget constraint
# ==========================================================================================

# --------- First, obtain the weights so that I can define A_hat, B_hat, C_hat -------
N = len(returns.columns)
e = np.ones(N)
V_inverse = np.linalg.inv(v_hat)
mu_vec = mu_hat.values   # add this line


A_hat = e.T @ V_inverse @ e
B_hat = mu_hat.values.T @ V_inverse @ e
C_hat = mu_hat.values.T @ V_inverse @ mu_hat.values

w_G = (V_inverse @ e) / (A_hat) # empirical global min-variance weights 
mu_G = B_hat/ A_hat # its expected return 
sigma_G = np.sqrt(1.0 / A_hat) # its risk 

print( "A_hat, B_hat, C_hat = ",A_hat,B_hat,C_hat)
print("w_G =", w_G)
print("sum(w_G) =", w_G.sum()) # should be 1 
print("mu_G =", mu_G)
print("sigma_G =", sigma_G)

# sanity checks
assert A_hat > 0 and C_hat > 0
assert A_hat * C_hat - B_hat**2 > 0
print("Markowitz scalar checks: PASS")

# ======================= PART C) =========================================================
# Explicit w_m computation and frontier plot

import matplotlib.pyplot as plt

# Compute w_m for a grid of target returns
m_grid = np.linspace(-0.001, 0.004, 500)
denom = A_hat * C_hat - B_hat**2

w_m_grid = np.zeros((len(m_grid), N))
sigma_m_grid = np.zeros(len(m_grid))

for j, m in enumerate(m_grid):
    w_m = ((C_hat - m*B_hat) / denom) * (V_inverse @ e) \
        + ((m*A_hat - B_hat) / denom) * (V_inverse @ mu_vec)
    w_m_grid[j] = w_m
    sigma_m_grid[j] = np.sqrt(w_m @ v_hat @ w_m)

# Cross-check: the closed-form frontier should match
sigma_closed = np.sqrt(np.clip(
    (A_hat*m_grid**2 - 2*B_hat*m_grid + C_hat) / denom, 0, None))
print("Max discrepancy between w_m and closed-form sigma:",
      np.max(np.abs(sigma_m_grid - sigma_closed)))

# Pick out a few specific targets for reporting
for m in [0.0, 0.0005, 0.001, 0.0015]:
    idx = np.argmin(np.abs(m_grid - m))
    w_m = w_m_grid[idx]
    print(f"\nm = {m:.4f}")
    print(f"  w_m = {np.round(w_m, 4)}")
    print(f"  sum(w_m) = {w_m.sum():.6f}")
    print(f"  sigma_P = {sigma_m_grid[idx]:.6f}")
    print(f"  mu_P    = {w_m @ mu_vec:.6f}")

# Plot frontier with individual stocks and GMV
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(sigma_m_grid, m_grid, label=r"Markowitz frontier $\hat\sigma_P(m)$",
        color="steelblue", linewidth=2)

for ticker in returns.columns:
    ax.scatter(sigma_hat[ticker], mu_hat[ticker], zorder=5, s=60)
    ax.annotate(ticker, (sigma_hat[ticker], mu_hat[ticker]),
                textcoords="offset points", xytext=(6, 4), fontsize=10)

ax.scatter(sigma_G, mu_G, color="red", marker="*", s=250, zorder=6,
           label=f"GMV ($\\hat\\sigma_G = {sigma_G:.4f}$)")

# Mark the two target returns from Part (d) — useful for continuity
for m_target, name in [(0.00055, "m1"), (0.00119, "m2")]:
    idx = np.argmin(np.abs(m_grid - m_target))
    ax.scatter(sigma_m_grid[idx], m_grid[idx], color="black",
               marker="x", s=80, zorder=7,
               label=f"{name} (m={m_target})")

ax.set_xlabel(r"risk $\hat\sigma_P$")
ax.set_ylabel(r"expected return $m$")
ax.set_title("Empirical Markowitz frontier (Oslo stocks, 2023–2025)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================================================================================
