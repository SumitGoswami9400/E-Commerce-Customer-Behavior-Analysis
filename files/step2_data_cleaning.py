"""
STEP 2: Data Cleaning with Pandas
- Missing values handle karo
- Duplicates remove karo
- Data types fix karo
- Outliers check karo
"""

import pandas as pd
import numpy as np

# ─── Load raw data ────────────────────────────────────────
df = pd.read_csv('ecommerce_raw.csv')

print("=" * 55)
print("BEFORE CLEANING")
print("=" * 55)
print(f"Shape            : {df.shape}")
print(f"Duplicate rows   : {df.duplicated().sum()}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ─── 1. Remove Duplicates ─────────────────────────────────
df.drop_duplicates(inplace=True)
print(f"\n✅ Duplicates removed. New shape: {df.shape}")

# ─── 2. Fix Data Types ────────────────────────────────────
df['order_date']    = pd.to_datetime(df['order_date'])
df['unit_price']    = df['unit_price'].astype(float)
df['total_price']   = df['total_price'].astype(float)
df['quantity']      = df['quantity'].astype(int)
df['discount_pct']  = df['discount_pct'].astype(int)
print("✅ Data types fixed.")

# ─── 3. Handle Missing Values ─────────────────────────────
# Rating: median se fill karo (outliers se safe)
median_rating = df['rating'].median()
df['rating'].fillna(median_rating, inplace=True)

# Payment method: mode se fill karo
mode_payment = df['payment_method'].mode()[0]
df['payment_method'].fillna(mode_payment, inplace=True)

print(f"✅ Missing 'rating' filled with median: {median_rating}")
print(f"✅ Missing 'payment_method' filled with mode: {mode_payment}")

# ─── 4. Feature Engineering ───────────────────────────────
# Naye useful columns banao

# Year, Month, Day of Week extract karo
df['year']       = df['order_date'].dt.year
df['month']      = df['order_date'].dt.month
df['month_name'] = df['order_date'].dt.strftime('%b')
df['day_of_week']= df['order_date'].dt.day_name()
df['quarter']    = df['order_date'].dt.quarter

# Revenue = total_price (already calculated)
# Add 'revenue_band' column
df['revenue_band'] = pd.cut(
    df['total_price'],
    bins=[0, 500, 2000, 10000, float('inf')],
    labels=['Low (<500)', 'Medium (500-2K)', 'High (2K-10K)', 'Premium (>10K)']
)

# ─── 5. Outlier Check ─────────────────────────────────────
Q1 = df['total_price'].quantile(0.25)
Q3 = df['total_price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['total_price'] < lower_bound) | (df['total_price'] > upper_bound)]
print(f"\n📊 Outliers in total_price: {len(outliers)} rows")
print(f"   (These are valid high-value purchases — kept as is)")

# ─── 6. Validate Data ─────────────────────────────────────
assert df.isnull().sum().sum() == 0, "❌ Still has missing values!"
assert df.duplicated().sum() == 0,   "❌ Still has duplicates!"
assert (df['total_price'] >= 0).all(), "❌ Negative prices found!"

# ─── 7. Save Cleaned Data ─────────────────────────────────
df.to_csv('ecommerce_cleaned.csv', index=False)

print("\n" + "=" * 55)
print("AFTER CLEANING")
print("=" * 55)
print(f"Shape            : {df.shape}")
print(f"Missing values   : {df.isnull().sum().sum()}")
print(f"Duplicates       : {df.duplicated().sum()}")
print(f"\nColumn info:")
print(df.dtypes)
print("\n✅ Cleaned data saved: ecommerce_cleaned.csv")
