"""
STEP 5: Revenue Analysis & Business Insights
- Seasonal trends identify karo
- Best performing categories
- Discount impact analysis
- Return rate analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style='whitegrid')
df = pd.read_csv('ecommerce_cleaned.csv', parse_dates=['order_date'])

# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: Quarterly Revenue Comparison
# ═══════════════════════════════════════════════════════════
quarterly = df.groupby(['year', 'quarter'])['total_price'].sum().reset_index()
quarterly['label'] = quarterly['year'].astype(str) + ' Q' + quarterly['quarter'].astype(str)

fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = ['#3498DB' if y == 2022 else '#E74C3C' for y in quarterly['year']]
bars = ax.bar(quarterly['label'], quarterly['total_price'] / 1e6, color=bar_colors, width=0.6, edgecolor='white')

for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f'₹{bar.get_height():.1f}M', ha='center', fontsize=8, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#3498DB', label='2022'), Patch(facecolor='#E74C3C', label='2023')]
ax.legend(handles=legend_elements)
ax.set_title('Quarterly Revenue — 2022 vs 2023', fontweight='bold', pad=12)
ax.set_ylabel('Revenue (₹ Millions)')
ax.set_xlabel('Quarter')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('chart9_quarterly.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: Discount Impact on Revenue
# ═══════════════════════════════════════════════════════════
disc_analysis = df.groupby('discount_pct').agg(
    avg_order_value = ('total_price', 'mean'),
    order_count     = ('transaction_id', 'count'),
    total_revenue   = ('total_price', 'sum')
).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.bar(disc_analysis['discount_pct'], disc_analysis['order_count'],
        color='#BDC3C7', alpha=0.8, label='Order Count', width=3)
ax2.plot(disc_analysis['discount_pct'], disc_analysis['avg_order_value'],
         color='#E74C3C', marker='o', linewidth=2, label='Avg Order Value (₹)')

ax1.set_xlabel('Discount %')
ax1.set_ylabel('Number of Orders', color='gray')
ax2.set_ylabel('Avg Order Value (₹)', color='#E74C3C')
ax2.tick_params(axis='y', labelcolor='#E74C3C')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.title('Discount % vs Order Volume & Average Order Value', fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('chart10_discount_impact.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: Return Rate by Category
# ═══════════════════════════════════════════════════════════
return_rate = df.groupby('category').apply(
    lambda x: (x['returned'] == 'Yes').sum() / len(x) * 100
).reset_index(name='return_rate_pct').sort_values('return_rate_pct', ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
colors_ret = ['#E74C3C' if r > 9 else '#F39C12' if r > 7 else '#2ECC71'
              for r in return_rate['return_rate_pct']]
ax.bar(return_rate['category'], return_rate['return_rate_pct'],
       color=colors_ret, edgecolor='white')
ax.axhline(y=8, color='red', linestyle='--', linewidth=1.2, label='Avg return rate (8%)')
for i, row in return_rate.iterrows():
    ax.text(i if False else list(return_rate['category']).index(row['category']),
            row['return_rate_pct'] + 0.1,
            f"{row['return_rate_pct']:.1f}%", ha='center', fontsize=9)

ax.set_title('Return Rate by Category', fontweight='bold', pad=12)
ax.set_ylabel('Return Rate (%)')
ax.legend()
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('chart11_return_rate.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# ANALYSIS 4: Day-of-Week Revenue Pattern
# ═══════════════════════════════════════════════════════════
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_rev   = df.groupby('day_of_week')['total_price'].sum().reindex(day_order)

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors_dow = ['#E74C3C' if d in ['Saturday','Sunday'] else '#5DADE2' for d in day_order]
ax.bar(day_rev.index, day_rev.values / 1e6, color=bar_colors_dow, edgecolor='white')
ax.set_title('Revenue by Day of Week\n(Red = Weekend)', fontweight='bold', pad=12)
ax.set_ylabel('Revenue (₹ Millions)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))
plt.tight_layout()
plt.savefig('chart12_day_of_week.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY STATS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("EXECUTIVE SUMMARY")
print("=" * 55)
total_rev  = df['total_price'].sum()
top3_cats  = df.groupby('category')['total_price'].sum().nlargest(3)
top3_share = top3_cats.sum() / total_rev * 100
print(f"💰 Total Revenue          : ₹{total_rev/1e7:.2f} Cr")
print(f"📦 Total Transactions     : {len(df):,}")
print(f"👥 Unique Customers       : {df['customer_id'].nunique():,}")
print(f"🛒 Avg Order Value        : ₹{df['total_price'].mean():,.0f}")
print(f"⭐ Avg Customer Rating    : {df['rating'].mean():.2f} / 5")
print(f"↩️  Overall Return Rate    : {(df['returned']=='Yes').mean()*100:.1f}%")
print(f"\n🔝 Top 3 Categories drive {top3_share:.1f}% of revenue:")
for cat, rev in top3_cats.items():
    print(f"   • {cat:20s} ₹{rev/1e6:.1f}M ({rev/total_rev*100:.1f}%)")
print("\n✅ All revenue charts saved!")
