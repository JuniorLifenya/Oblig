# ======================= PART 2) Problem 2 =============================================
# One-period risk-neutral pricing

import numpy as np

# ---------- Part 2(a): risk-neutral probability and no-arbitrage condition ----------
def risk_neutral_prob(S0, Su, Sd, r):
    """Compute the risk-neutral probability q on the two-state model."""
    return (S0 * (1 + r) - Sd) / (Su - Sd)

def is_arbitrage_free(S0, Su, Sd, r):
    """Return True if Sd < S0*(1+r) < Su holds."""
    return Sd < S0 * (1 + r) < Su

# ---------- Part 2(b): state claim prices ----------
def state_prices(q, r):
    """Prices of the two Arrow-Debreu securities: pi_u and pi_d."""
    return q / (1 + r), (1 - q) / (1 + r)

# ---------- Part 2(c): European call and put prices ----------
def call_price(q, r, Su, Sd, K):
    """No-arbitrage price of a European call with strike K."""
    payoff_u = max(Su - K, 0)
    payoff_d = max(Sd - K, 0)
    return (q * payoff_u + (1 - q) * payoff_d) / (1 + r)

def put_price(q, r, Su, Sd, K):
    """No-arbitrage price of a European put with strike K."""
    payoff_u = max(K - Su, 0)
    payoff_d = max(K - Sd, 0)
    return (q * payoff_u + (1 - q) * payoff_d) / (1 + r)

# ---------- Specific numbers from the assignment ----------
r  = 1/20
S0 = 80
Su = 100
Sd = 60
K_C = 95
K_P = 65

print("=" * 60)
print("PROBLEM 2")
print("=" * 60)

print(f"\nInputs: r = {r}, S0 = {S0}, Su = {Su}, Sd = {Sd}")

# Part (a)
print("\n--- Part (a) ---")
print(f"S0(1+r) = {S0*(1+r):.4f}")
print(f"No-arbitrage condition Sd < S0(1+r) < Su: "
      f"{is_arbitrage_free(S0, Su, Sd, r)}")

q = risk_neutral_prob(S0, Su, Sd, r)
print(f"Risk-neutral q      = {q:.4f}")
print(f"Risk-neutral 1-q    = {1-q:.4f}")

# Part (b)
print("\n--- Part (b) ---")
pi_u, pi_d = state_prices(q, r)
print(f"pi_u (price of X_u) = {pi_u:.6f}")
print(f"pi_d (price of X_d) = {pi_d:.6f}")
print(f"pi_u + pi_d         = {pi_u + pi_d:.6f}  (= 1/(1+r) = {1/(1+r):.6f})")

# Part (c)
print("\n--- Part (c) ---")
C_payoff = (max(Su-K_C, 0), max(Sd-K_C, 0))
P_payoff = (max(K_P-Su, 0), max(K_P-Sd, 0))
print(f"Call payoff vector (K={K_C}): {C_payoff}")
print(f"Put  payoff vector (K={K_P}): {P_payoff}")

C0 = call_price(q, r, Su, Sd, K_C)
P0 = put_price(q, r, Su, Sd, K_P)
print(f"Call price C0 = {C0:.6f}")
print(f"Put  price P0 = {P0:.6f}")

# Sanity check: put-call parity
K = 65
C_K = call_price(q, r, Su, Sd, K)
P_K = put_price(q, r, Su, Sd, K)
print(f"\nSanity check (put-call parity, K={K}):")
print(f"  C_K - P_K          = {C_K - P_K:.6f}")
print(f"  S0 - K/(1+r)       = {S0 - K/(1+r):.6f}")
assert np.isclose(C_K - P_K, S0 - K/(1+r))
print("  PASS")

# Sanity check: sum of state prices = bond price
assert np.isclose(pi_u + pi_d, 1/(1+r))
print(f"  state prices sum to 1/(1+r): PASS")