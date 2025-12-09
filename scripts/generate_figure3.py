#!/usr/bin/env python3
"""
Generate Figure 3: Sample Efficiency Analysis
Shows relationship between samples per feature and R² performance
"""

import matplotlib.pyplot as plt
import numpy as np

def generate_figure3():
    """Create scatter plot showing sample efficiency vs performance"""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Data points from validation_results.json (VERIFIED)
    systems_data = [
        ('IBM 4Q', 5.0, 0.764, 'blue', 'o'),      # 50 samples / 10 features
        ('IBM 156Q', 1.28, 0.723, 'orange', 's'), # 200 samples / 156 features
        ('Rigetti 9Q', 0.19, 0.959, 'green', '^'), # 640 samples / 3375 features
    ]

    # Plot data points
    for name, samp_per_feat, r2, color, marker in systems_data:
        ax.scatter(samp_per_feat, r2, s=300, c=color, marker=marker,
                   alpha=0.7, edgecolors='black', linewidth=2, label=name)
        # Add labels with offset
        ax.annotate(name, (samp_per_feat, r2), xytext=(10, 10),
                    textcoords='offset points', fontsize=11, fontweight='bold')

    # Add theoretical thresholds
    ax.axvline(x=5.0, color='green', linestyle='--', linewidth=2,
               label='Healthy Threshold (>5)', alpha=0.7)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2,
               label='Critical Threshold (<0.1)', alpha=0.7)

    # Shaded regions for interpretation
    ax.axvspan(5.0, 15, alpha=0.1, color='green', label='Safe Zone')
    ax.axvspan(0.01, 0.1, alpha=0.1, color='red', label='Critical Zone')
    ax.axvspan(0.1, 5.0, alpha=0.05, color='yellow', label='Intermediate Zone')

    # Logarithmic scale for x-axis
    ax.set_xscale('log')
    ax.set_xlabel('Samples per Feature (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
    ax.set_title('Sample Efficiency vs. Performance: QRC Systems Comparison',
                 fontsize=14, fontweight='bold', pad=20)

    # Set axis limits
    ax.set_xlim([0.01, 20])
    ax.set_ylim([-0.1, 1.0])

    # Grid and styling
    ax.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

    # Add annotation box with key insight
    textstr = 'Rigetti 9Q achieves 96% variance\nexplanation with 26× fewer\nsamples per feature than IBM 4Q'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig('figures/figure3_sample_efficiency.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure3_sample_efficiency.pdf', bbox_inches='tight')

    print("Figure 3 generated successfully!")
    print("  - PNG: figures/figure3_sample_efficiency.png")
    print("  - PDF: figures/figure3_sample_efficiency.pdf")

    plt.close()

if __name__ == '__main__':
    generate_figure3()
