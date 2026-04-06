"""
STEP 3: Exploratory Data Analysis (EDA)
- Category-wise revenue
- Monthly trends
- Top cities
- Payment methods
- Rating distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ─── Style setup ──────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi'    : 120,
    'font.family'   : 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

df = pd.read_csv('ecommerce_cleaned.csv', parse_dates=['order_date'])

# ═══════════════════════════════════════════════════════════
# FIGURE 1: Category-wise Revenue (Bar Chart)
# ═══════════════════════════════════════════════════════════
cat_rev = df.groupby('category')['total_price'].sum().sort_values(ascending=False)
top3    = cat_rev.index[:3].tolist()

fig, ax = plt.subplots(figsize=(10, 5))
colors  = ['#E74C3C' if c in top3 else '#5DADE2' for c in cat_rev.index]
bars    = ax.bar(cat_rev.index, cat_rev.values / 1e6, color=colors, edgecolor='white', linewidth=0.6)

# Value labels on bars
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'₹{bar.get_height():.1f}M',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('Category-wise Total Revenue\n(Red = Top 3 categories)', fontweight='bold', pad=12)
ax.set_xlabel('Product Category')
ax.set_ylabel('Total Revenue (₹ Millions)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('chart1_category_revenue.png', bbox_inches='tight')
plt.show()

rev_pct = cat_rev[:3].sum() / cat_rev.sum() * 100
print(f"\n📌 Top 3 categories contribute {rev_pct:.1f}% of total revenue")
print(f"   They are: {', '.join(top3)}")

# ═══════════════════════════════════════════════════════════
# FIGURE 2: Monthly Revenue Trend (Line Chart)
# ═══════════════════════════════════════════════════════════
monthly = df.groupby(['year', 'month'])['total_price'].sum().reset_index()
monthly['period'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
monthly.sort_values('period', inplace=True)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly['period'], monthly['total_price'] / 1e6,
        marker='o', color='#2E86C1', linewidth=2, markersize=5)
ax.fill_between(monthly['period'], monthly['total_price'] / 1e6, alpha=0.12, color='#2E86C1')

# Peak highlight
peak_idx  = monthly['total_price'].idxmax()
peak_row  = monthly.loc[peak_idx]
ax.annotate(f"Peak\n₹{peak_row['total_price']/1e6:.1f}M",
            xy=(peak_row['period'], peak_row['total_price'] / 1e6),
            xytext=(30, 20), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=9, color='red')

ax.set_title('Monthly Revenue Trend (2022-2023)', fontweight='bold', pad=12)
ax.set_xlabel('Month')
ax.set_ylabel('Revenue (₹ Millions)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.1f}M'))
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('chart2_monthly_trend.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# FIGURE 3: Top Cities by Revenue (Horizontal Bar)
# ═══════════════════════════════════════════════════════════
city_rev = df.groupby('city')['total_price'].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors_city = sns.color_palette('Blues_d', len(city_rev))
ax.barh(city_rev.index, city_rev.values / 1e6, color=colors_city, edgecolor='white')

for i, v in enumerate(city_rev.values):
    ax.text(v / 1e6 + 0.1, i, f'₹{v/1e6:.1f}M', va='center', fontsize=9)

ax.set_title('Revenue by City', fontweight='bold', pad=12)
ax.set_xlabel('Total Revenue (₹ Millions)')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:.0f}M'))
plt.tight_layout()
plt.savefig('chart3_city_revenue.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# FIGURE 4: Payment Method Distribution (Pie Chart)
# ═══════════════════════════════════════════════════════════
pay_counts = df['payment_method'].value_counts()
explode    = [0.04] * len(pay_counts)
colors_pie = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']

fig, ax = plt.subplots(figsize=(7, 6))
wedges, texts, autotexts = ax.pie(
    pay_counts.values, labels=pay_counts.index,
    autopct='%1.1f%%', explode=explode,
    colors=colors_pie, startangle=140,
    pctdistance=0.82, labeldistance=1.1)

for at in autotexts:
    at.set_fontsize(9)
    at.set_color('white')
    at.set_fontweight('bold')

ax.set_title('Payment Method Distribution', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('chart4_payment_methods.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# FIGURE 5: Rating Distribution (Histogram + KDE)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(df['rating'], bins=20, kde=True, color='#8E44AD',
             line_kws={'linewidth': 2}, ax=ax)
ax.axvline(df['rating'].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df['rating'].mean():.2f}")
ax.axvline(df['rating'].median(), color='orange', linestyle='--', linewidth=1.5, label=f"Median: {df['rating'].median():.2f}")
ax.legend()
ax.set_title('Customer Rating Distribution', fontweight='bold', pad=12)
ax.set_xlabel('Rating (1-5)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('chart5_rating_dist.png', bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════
# FIGURE 6: Heatmap — Category Revenue by Month
# ═══════════════════════════════════════════════════════════
pivot = df.pivot_table(index='month_name', columns='category',
                       values='total_price', aggfunc='sum')
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot.reindex([m for m in month_order if m in pivot.index])

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot / 1e6, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Revenue (₹M)'})
ax.set_title('Monthly Revenue Heatmap by Category (₹ Millions)', fontweight='bold', pad=12)
ax.set_xlabel('')
ax.set_ylabel('Month')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('chart6_heatmap.png', bbox_inches='tight')
plt.show()

print("\n✅ All 6 charts saved as PNG files!")
