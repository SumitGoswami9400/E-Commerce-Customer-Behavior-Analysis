"""
STEP 1: E-Commerce Dataset Generate Karna
- 50,000 customer transactions create karega
- Real-world jaisi data structure hogi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Seed set karo taaki results reproduce ho sakein
np.random.seed(42)
random.seed(42)

# ─── Constants ───────────────────────────────────────────
NUM_RECORDS      = 50000
NUM_CUSTOMERS    = 5000
START_DATE       = datetime(2022, 1, 1)
END_DATE         = datetime(2023, 12, 31)

CATEGORIES       = ['Electronics', 'Fashion', 'Home & Kitchen', 'Sports', 'Books', 'Beauty', 'Toys']
CATEGORY_WEIGHTS = [0.25, 0.20, 0.18, 0.12, 0.10, 0.10, 0.05]  # Electronics sabse zyada bika

PAYMENT_METHODS  = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Wallet']
CITIES           = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata', 'Ahmedabad']

# ─── Price ranges per category ────────────────────────────
PRICE_RANGE = {
    'Electronics'   : (500,  80000),
    'Fashion'       : (200,   5000),
    'Home & Kitchen': (150,  15000),
    'Sports'        : (300,  20000),
    'Books'         : (100,   1500),
    'Beauty'        : (100,   3000),
    'Toys'          : (200,   5000),
}

# ─── Date helper ─────────────────────────────────────────
def random_date(start, end):
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

# ─── Generate customer IDs (kuch repeat honge → repeat buyers) ──
customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, NUM_CUSTOMERS + 1)]

# ─── Build dataset row by row ─────────────────────────────
records = []
for i in range(1, NUM_RECORDS + 1):
    category     = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
    min_p, max_p = PRICE_RANGE[category]
    unit_price   = round(np.random.uniform(min_p, max_p), 2)
    quantity     = np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.10, 0.06, 0.04])
    discount_pct = np.random.choice([0, 5, 10, 15, 20, 25, 30], p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05])
    total_before = unit_price * quantity
    discount_amt = round(total_before * discount_pct / 100, 2)
    total_price  = round(total_before - discount_amt, 2)

    records.append({
        'transaction_id' : f"TXN_{str(i).zfill(6)}",
        'customer_id'    : random.choice(customer_ids),
        'order_date'     : random_date(START_DATE, END_DATE).strftime('%Y-%m-%d'),
        'category'       : category,
        'product_name'   : f"{category}_Product_{random.randint(1, 200)}",
        'unit_price'     : unit_price,
        'quantity'       : quantity,
        'discount_pct'   : discount_pct,
        'discount_amount': discount_amt,
        'total_price'    : total_price,
        'payment_method' : random.choice(PAYMENT_METHODS),
        'city'           : random.choice(CITIES),
        'rating'         : round(np.random.uniform(1, 5), 1),
        'returned'       : np.random.choice(['Yes', 'No'], p=[0.08, 0.92]),  # 8% return rate
    })

df = pd.DataFrame(records)

# ─── Intentionally add some dirty data (cleaning practice ke liye) ──
# 200 rows mein missing values daalo
missing_idx = np.random.choice(df.index, 200, replace=False)
df.loc[missing_idx[:100], 'rating']         = np.nan
df.loc[missing_idx[100:], 'payment_method'] = np.nan

# 50 rows mein duplicates daalo
dup_idx = np.random.choice(df.index, 50, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# ─── Save ─────────────────────────────────────────────────
df.to_csv('ecommerce_raw.csv', index=False)

print(f"✅ Dataset saved: ecommerce_raw.csv")
print(f"   Total rows    : {len(df):,}")
print(f"   Total columns : {len(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())
