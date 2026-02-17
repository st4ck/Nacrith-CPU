#!/usr/bin/env python3
"""
Generate charts from text-only benchmark results.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Dark theme colors
BG_COLOR = '#0D1117'
TEXT_COLOR = '#E6EDF3'
GRID_COLOR = '#30363D'
C_GZIP = '#6C8EBF'
C_LZMA = '#82B366'
C_NACRITH = '#E04040'

# Configure matplotlib for dark theme
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = BG_COLOR
plt.rcParams['axes.facecolor'] = BG_COLOR
plt.rcParams['axes.edgecolor'] = GRID_COLOR
plt.rcParams['grid.color'] = GRID_COLOR
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Load results
with open('benchmark_text_results.json', 'r') as f:
    results = json.load(f)

# Sort by original size
results = sorted(results, key=lambda x: x['original_size'])

# Extract data
filenames = [r['filename'].replace('.txt', '') for r in results]
orig_sizes = [r['original_size'] for r in results]
gzip_sizes = [r['compressors']['gzip']['size'] for r in results]
lzma_sizes = [r['compressors']['lzma']['size'] for r in results]
nacrith_sizes = [r['compressors']['nacrith']['size'] for r in results]

gzip_ratios = [r['compressors']['gzip']['ratio'] * 100 for r in results]
lzma_ratios = [r['compressors']['lzma']['ratio'] * 100 for r in results]
nacrith_ratios = [r['compressors']['nacrith']['ratio'] * 100 for r in results]

gzip_bpb = [r['compressors']['gzip']['bpb'] for r in results]
lzma_bpb = [r['compressors']['lzma']['bpb'] for r in results]
nacrith_bpb = [r['compressors']['nacrith']['bpb'] for r in results]

# Create output directory
os.makedirs('assets', exist_ok=True)

# Shorten long names for charts
short_names = []
for name in filenames:
    if len(name) > 20:
        # Take first words that fit
        words = name.split()
        short = words[0]
        for w in words[1:]:
            if len(short + ' ' + w) <= 20:
                short += ' ' + w
            else:
                break
        short_names.append(short + '...')
    else:
        short_names.append(name)

# Format file sizes for labels
size_labels = []
for size in orig_sizes:
    if size < 1024 * 100:
        size_labels.append(f"{size/1024:.0f} KB")
    elif size < 1024 * 1024:
        size_labels.append(f"{size/1024:.0f} KB")
    else:
        size_labels.append(f"{size/(1024*1024):.1f} MB")

# Combined labels: filename + size
combined_labels = [f"{name}\n({size})" for name, size in zip(short_names, size_labels)]

# ==================================================================
# Chart 1: Compression Ratio by File (%)
# ==================================================================
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(filenames))
width = 0.25

bars1 = ax.bar(x - width, gzip_ratios, width, label='gzip -9', color=C_GZIP, alpha=0.9)
bars2 = ax.bar(x, lzma_ratios, width, label='lzma -9', color=C_LZMA, alpha=0.9)
bars3 = ax.bar(x + width, nacrith_ratios, width, label='Nacrith v7.6', color=C_NACRITH, alpha=0.9)

ax.set_xlabel('Book Title (Original Size)', fontsize=11, weight='bold')
ax.set_ylabel('Compression Ratio (%)', fontsize=11, weight='bold')
ax.set_title('Compression Ratio on Classic Literature\n(Lower is Better)', fontsize=13, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(combined_labels, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(gzip_ratios) * 1.1)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=7, weight='bold')

plt.tight_layout()
plt.savefig('assets/compression_ratio_by_file.png', dpi=150, facecolor=BG_COLOR)
print("Generated: assets/compression_ratio_by_file.png")
plt.close()

# ==================================================================
# Chart 2: Bits Per Byte by File
# ==================================================================
fig, ax = plt.subplots(figsize=(14, 7))

bars1 = ax.bar(x - width, gzip_bpb, width, label='gzip -9', color=C_GZIP, alpha=0.9)
bars2 = ax.bar(x, lzma_bpb, width, label='lzma -9', color=C_LZMA, alpha=0.9)
bars3 = ax.bar(x + width, nacrith_bpb, width, label='Nacrith v7.6', color=C_NACRITH, alpha=0.9)

ax.set_xlabel('Book Title (Original Size)', fontsize=11, weight='bold')
ax.set_ylabel('Bits Per Byte', fontsize=11, weight='bold')
ax.set_title('Compression Efficiency: Bits Per Byte\n(Lower is Better)', fontsize=13, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(combined_labels, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(gzip_bpb) * 1.1)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7, weight='bold')

plt.tight_layout()
plt.savefig('assets/bits_per_byte_by_file.png', dpi=150, facecolor=BG_COLOR)
print("Generated: assets/bits_per_byte_by_file.png")
plt.close()

# ==================================================================
# Chart 3: Compression Improvement vs File Size (scatter)
# ==================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Calculate improvement percentage (Nacrith vs lzma)
improvements = [(lzma - nacrith) / lzma * 100
                for lzma, nacrith in zip(lzma_sizes, nacrith_sizes)]

# Convert sizes to KB for plotting
sizes_kb = [s / 1024 for s in orig_sizes]

# Scatter plot with file names
scatter = ax.scatter(sizes_kb, improvements, s=200, c=C_NACRITH, alpha=0.7, edgecolors='white', linewidth=1.5)

# Add labels for each point
for i, (x, y, name) in enumerate(zip(sizes_kb, improvements, short_names)):
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10),
                ha='center', fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_COLOR, edgecolor=C_NACRITH, alpha=0.8))

ax.set_xlabel('File Size (KB)', fontsize=11, weight='bold')
ax.set_ylabel('Improvement over lzma (%)', fontsize=11, weight='bold')
ax.set_title('Nacrith v7.6 Improvement vs File Size', fontsize=13, weight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Add reference line at 0%
ax.axhline(y=0, color=C_LZMA, linestyle='--', linewidth=1, alpha=0.5, label='Equal to lzma')

# Add average line
avg_improvement = np.mean(improvements)
ax.axhline(y=avg_improvement, color='white', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Average: {avg_improvement:.1f}%')

ax.legend(loc='lower left', framealpha=0.9)
ax.set_ylim(min(improvements) * 0.8, max(improvements) * 1.15)

plt.tight_layout()
plt.savefig('assets/improvement_vs_size.png', dpi=150, facecolor=BG_COLOR)
print("Generated: assets/improvement_vs_size.png")
plt.close()

# ==================================================================
# Chart 4: Compressed Size Comparison (stacked area / line chart)
# ==================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Convert to KB
orig_kb = [s / 1024 for s in orig_sizes]
gzip_kb = [s / 1024 for s in gzip_sizes]
lzma_kb = [s / 1024 for s in lzma_sizes]
nacrith_kb = [s / 1024 for s in nacrith_sizes]

x_pos = np.arange(len(filenames))

ax.plot(x_pos, orig_kb, marker='s', linewidth=2, markersize=8,
        label='Original', color='white', alpha=0.5, linestyle='--')
ax.plot(x_pos, gzip_kb, marker='o', linewidth=2.5, markersize=8,
        label='gzip -9', color=C_GZIP, alpha=0.9)
ax.plot(x_pos, lzma_kb, marker='^', linewidth=2.5, markersize=8,
        label='lzma -9', color=C_LZMA, alpha=0.9)
ax.plot(x_pos, nacrith_kb, marker='D', linewidth=3, markersize=8,
        label='Nacrith v7.6', color=C_NACRITH, alpha=0.9)

ax.set_xlabel('Book Title', fontsize=11, weight='bold')
ax.set_ylabel('Size (KB)', fontsize=11, weight='bold')
ax.set_title('Compressed Size Comparison Across Files', fontsize=13, weight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('assets/size_comparison_line.png', dpi=150, facecolor=BG_COLOR)
print("Generated: assets/size_comparison_line.png")
plt.close()

# ==================================================================
# Chart 5: Average Performance Summary (single bar chart)
# ==================================================================
fig, ax = plt.subplots(figsize=(10, 6))

avg_gzip_ratio = np.mean(gzip_ratios)
avg_lzma_ratio = np.mean(lzma_ratios)
avg_nacrith_ratio = np.mean(nacrith_ratios)

avg_gzip_bpb = np.mean(gzip_bpb)
avg_lzma_bpb = np.mean(lzma_bpb)
avg_nacrith_bpb = np.mean(nacrith_bpb)

categories = ['Compression Ratio (%)', 'Bits Per Byte']
gzip_vals = [avg_gzip_ratio, avg_gzip_bpb]
lzma_vals = [avg_lzma_ratio, avg_lzma_bpb]
nacrith_vals = [avg_nacrith_ratio, avg_nacrith_bpb]

x_cat = np.arange(len(categories))
width = 0.25

bars1 = ax.bar(x_cat - width, gzip_vals, width, label='gzip -9', color=C_GZIP, alpha=0.9)
bars2 = ax.bar(x_cat, lzma_vals, width, label='lzma -9', color=C_LZMA, alpha=0.9)
bars3 = ax.bar(x_cat + width, nacrith_vals, width, label='Nacrith v7.6', color=C_NACRITH, alpha=0.9)

ax.set_ylabel('Value (Lower is Better)', fontsize=11, weight='bold')
ax.set_title('Average Performance Across All 10 Books', fontsize=13, weight='bold', pad=20)
ax.set_xticks(x_cat)
ax.set_xticklabels(categories, fontsize=11, weight='bold')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('assets/average_performance.png', dpi=150, facecolor=BG_COLOR)
print("Generated: assets/average_performance.png")
plt.close()

# ==================================================================
# Chart 6: Space Savings vs File Size
# ==================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Calculate space savings (bytes saved compared to original)
gzip_savings = [(orig - comp) / orig * 100 for orig, comp in zip(orig_sizes, gzip_sizes)]
lzma_savings = [(orig - comp) / orig * 100 for orig, comp in zip(orig_sizes, lzma_sizes)]
nacrith_savings = [(orig - comp) / orig * 100 for orig, comp in zip(orig_sizes, nacrith_sizes)]

x_pos = np.arange(len(filenames))

ax.plot(x_pos, gzip_savings, marker='o', linewidth=2.5, markersize=8,
        label='gzip -9', color=C_GZIP, alpha=0.9)
ax.plot(x_pos, lzma_savings, marker='^', linewidth=2.5, markersize=8,
        label='lzma -9', color=C_LZMA, alpha=0.9)
ax.plot(x_pos, nacrith_savings, marker='D', linewidth=3, markersize=8,
        label='Nacrith v7.6', color=C_NACRITH, alpha=0.9)

ax.set_xlabel('Book Title (by size)', fontsize=11, weight='bold')
ax.set_ylabel('Space Savings (%)', fontsize=11, weight='bold')
ax.set_title('Space Savings: How Much Smaller vs Original\n(Higher is Better)',
             fontsize=13, weight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Add value labels for Nacrith only (to avoid clutter)
for i, (x, y) in enumerate(zip(x_pos, nacrith_savings)):
    ax.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom',
            fontsize=7, weight='bold', color=C_NACRITH)

plt.tight_layout()
plt.savefig('assets/space_savings.png', dpi=150, facecolor=BG_COLOR)
print("Generated: assets/space_savings.png")
plt.close()

print("\nAll charts generated successfully!")
print("Charts saved in assets/ directory")
