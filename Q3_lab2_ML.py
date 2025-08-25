import pandas as pd
import statistics
import matplotlib.pyplot as plt

# ===========================
# A3: IRCTC Stock Price Analysis
# ===========================

# ✅ Your dataset file (CSV)
file = r"C:\Users\anite\Downloads\Lab Session Data(IRCTC Stock Price) (1).csv"
df = pd.read_csv(file)

# --- Debug step ---
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# --- Clean numeric columns ---
# Strip commas, %, ₹ symbols, and spaces before converting
df["Price"] = df["Price"].astype(str).str.replace(",", "").str.replace("₹", "").str.strip()
df["Chg%"] = df["Chg%"].astype(str).str.replace("%", "").str.strip()

# Convert to numeric
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Chg%"] = pd.to_numeric(df["Chg%"], errors="coerce")

# Drop rows where Price is missing
df = df.dropna(subset=["Price"])

# 1. Mean & Variance of Price
price_mean = statistics.mean(df["Price"])
price_variance = statistics.variance(df["Price"])
print("\nPopulation Mean (Price):", price_mean)
print("Population Variance (Price):", price_variance)

# 2. Mean of Price on Wednesdays
wed_prices = df[df["Day"] == "Wednesday"]["Price"]
if not wed_prices.empty:
    wed_mean = statistics.mean(wed_prices)
    print("\nWednesday Sample Mean (Price):", wed_mean)
    print("Comparison → Wednesday Mean vs Population Mean:", wed_mean, "vs", price_mean)
else:
    print("\nNo Wednesday data found.")

# 3. Mean of Price in April
apr_prices = df[df["Month"] == "Apr"]["Price"]
if not apr_prices.empty:
    apr_mean = statistics.mean(apr_prices)
    print("\nApril Sample Mean (Price):", apr_mean)
    print("Comparison → April Mean vs Population Mean:", apr_mean, "vs", price_mean)
else:
    print("\nNo April data found.")

# 4. Probability of making a loss (Chg% < 0)
loss_prob = (df["Chg%"] < 0).mean()
print("\nP(Loss) =", loss_prob)

# 5. Probability of making a profit on Wednesday
if (df["Day"] == "Wednesday").sum() > 0:
    profit_wed_prob = ((df["Day"] == "Wednesday") & (df["Chg%"] > 0)).sum() / (df["Day"] == "Wednesday").sum()
    print("P(Profit on Wednesday) =", profit_wed_prob)
else:
    print("No Wednesday data to compute profit probability.")

# 6. Conditional probability P(Profit | Wednesday)
prob_wed = (df["Day"] == "Wednesday").mean()
prob_profit_and_wed = ((df["Day"] == "Wednesday") & (df["Chg%"] > 0)).mean()
if prob_wed > 0:
    cond_prob = prob_profit_and_wed / prob_wed
    print("P(Profit | Wednesday) =", cond_prob)
else:
    print("No Wednesday data for conditional probability.")

# 7. Scatter Plot of Chg% vs Day of Week
plt.scatter(df["Day"], df["Chg%"])
plt.title("Chg% vs Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Chg%")
plt.show()

