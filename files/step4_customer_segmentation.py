"""
STEP 4: Customer Segmentation using RFM Analysis
- R = Recency  (kitne din pehle kharida)
- F = Frequency (kitni baar kharida)
- M = Monetary  (kitne paise kharch kiye)
Interview mein ye sikhana bahut impactful hota hai!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ecommerce_cleaned.csv', parse_dates=['order_date'])

# ─── Reference date = last transaction ke baad ka din ────
ref_date = df['order_date'].max() + pd.Timedelta(days=1)

# ─── Calculate RFM per customer ──────────────────────────
rfm = df.groupby('customer_id').agg(
    recency   = ('order_date',  lambda x: (ref_date - x.max()).days),
    frequency = ('transaction_id', 'count'),
    monetary  = ('total_price',    'sum')
).reset_index()

print("RFM table (first 5 rows):")
print(rfm.head())
print(f"\nTotal unique customers: {len(rfm):,}")

# ─── Score karo 1-5 (5 = best) ───────────────────────────
rfm['r_score'] = pd.qcut(rfm['recency'],   q=5, labels=[5,4,3,2,1])   # kam recency = better
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
rfm['m_score'] = pd.qcut(rfm['monetary'],  q=5, labels=[1,2,3,4,5])

rfm['r_score'] = rfm['r_score'].astype(int)
rfm['f_score'] = rfm['f_score'].astype(int)
rfm['m_score'] = rfm['m_score'].astype(int)
rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']

# ─── Segments define karo ────────────────────────────────
def segment_customer(row):
    score = row['rfm_score']
    r, f  = row['r_score'], row['f_score']
    if score >= 13:
        return 'Champions'
    elif score >= 10:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'Promising'
    elif r <= 2 and f >= 4:
        return 'At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost'
    else:
        return 'Potential Loyalists'

rfm['segment'] = rfm.apply(segment_customer, axis=1)

# ─── Segment summary ─────────────────────────────────────
seg_summary = rfm.groupby('segment').agg(
    customer_count = ('customer_id', 'count'),
    avg_recency    = ('recency',   'mean'),
    avg_frequency  = ('frequency', 'mean'),
    avg_monetary   = ('monetary',  'mean'),
    total_revenue  = ('monetary',  'sum')
).round(2).sort_values('total_revenue', ascending=False)

print("\n📊 Segment Summary:")
print(seg_summary.to_string())

# ─── FIGURE 7: Segment Distribution (Bar) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seg_counts = rfm['segment'].value_counts()
palette    = sns.color_palette('Set2', len(seg_counts))

# Bar chart — customer count
axes[0].bar(seg_counts.index, seg_counts.values, color=palette, edgecolor='white')
for i, (idx, val) in enumerate(seg_counts.items()):
    axes[0].text(i, val + 10, str(val), ha='center', fontsize=9, fontweight='bold')
axes[0].set_title('Customers per Segment', fontweight='bold')
axes[0].set_ylabel('Customer Count')
axes[0].tick_params(axis='x', rotation=20)

# Pie — revenue per segment
seg_rev = rfm.groupby('segment')['monetary'].sum()
axes[1].pie(seg_rev.values, labels=seg_rev.index,
            autopct='%1.1f%%', colors=palette,
            startangle=140, pctdistance=0.80)
axes[1].set_title('Revenue Share by Segment', fontweight='bold')

plt.tight_layout()
plt.savefig('chart7_segments.png', bbox_inches='tight')
plt.show()

# ─── FIGURE 8: RFM Scatter Plot ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
seg_colors = {
    'Champions'         : '#E74C3C',
    'Loyal Customers'   : '#3498DB',
    'Promising'         : '#2ECC71',
    'At Risk'           : '#E67E22',
    'Lost'              : '#95A5A6',
    'Potential Loyalists': '#9B59B6',
}
for seg, grp in rfm.groupby('segment'):
    ax.scatter(grp['frequency'], grp['monetary'] / 1000,
               label=seg, alpha=0.5, s=25, color=seg_colors.get(seg, 'gray'))

ax.set_title('RFM Scatter: Frequency vs Monetary Value', fontweight='bold', pad=12)
ax.set_xlabel('Purchase Frequency')
ax.set_ylabel('Total Spend (₹ Thousands)')
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('chart8_rfm_scatter.png', bbox_inches='tight')
plt.show()

# ─── Save RFM data ────────────────────────────────────────
rfm.to_csv('ecommerce_rfm.csv', index=False)
print("\n✅ RFM data saved: ecommerce_rfm.csv")

# ─── Business Insights ───────────────────────────────────
print("\n" + "=" * 55)
print("KEY BUSINESS INSIGHTS")
print("=" * 55)
champions = rfm[rfm['segment'] == 'Champions']
at_risk   = rfm[rfm['segment'] == 'At Risk']
print(f"🏆 Champions : {len(champions):,} customers, avg spend ₹{champions['monetary'].mean():,.0f}")
print(f"⚠️  At Risk   : {len(at_risk):,} customers — target with re-engagement campaigns!")
print(f"💡 Top 20% customers drive ~{rfm.nlargest(int(len(rfm)*0.2), 'monetary')['monetary'].sum() / rfm['monetary'].sum()*100:.0f}% revenue")
