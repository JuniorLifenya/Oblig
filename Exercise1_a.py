# First I will Import the OSLO stock files
import numpy as np
import pandas as pd

# ============= Firstly we load the adjusted Prices sheet ==========================
# The workbook has two sheets: "ReadMe" and "Adjusted Prices".
# First row of "Adjusted Prices" is a title.
# Second row contains column headers (Date, EQNR, DNB, TEL, NAS, NHY).
# Skp the title row and use the header now.
prices = pd.read_csv(
    "oslo-stocks-2023-2025.xlsx",
    sheet_name = "Adjusted Prices",
    header = 1, # row 2 in Excel = header (0-indexed row 1)
    index_col = "Date",
    parse_dates= True, 
)
# Drop any fully-empty rows (the sheet sometimes has a trailing blanks)
prices = prices.dropna(how = "All")
print ("Shape:", prices.shape)
print("Columns: ", list(prices.columns))
print("Date Range: ", prices.index[0].date(), "to", prices.index[-1].date())
print(prices.head())


# ============= Now Compute Daily Simple returns ====================================

# The formula is r_{i,k} = (S_i(t_k) - S_i(t_{k-1})) / S_i(t_{k-1})
returns = prices.pct_change().dropna(how = "All")
print("\nReturns shape: ", returns.shape) # Expect (750,5) now

# ================= Now Sample moments (per stock) ===================================

mu_hat = returns.mean()

