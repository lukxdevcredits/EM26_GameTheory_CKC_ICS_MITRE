#!/usr/bin/env python3
"""
Phase 5 - FINAL VISUALIZATION (ACADEMIC TAXONOMY)
Author: LBD
Date: 2025
Description: Generates figures for final report.
             - Converts raw SQLite simulation data into IEEE plots
             - Maps raw experiment names to 'strategies'
             - Exports high-dpi (300) .pngs
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np

# Suppress warnings to keep the console clean 
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------
# CONFIG & PATHS
# -----------------------
DB_FILE = 'phase2.db'
OUTPUT_DIR = './Phase2_Outputs'
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# -----------------------
# STYLE & TAXONOMY
# -----------------------
# Consistent colors/markers across all charts
STYLE_MAP = {
    'Hardened Def':   {'color': '#3E2723', 'marker': 's', 'label': 'Hardened Defense'}, # Square (Heavy/Rigid)
    'Insider Threat': {'color': '#8D6E63', 'marker': 'D', 'label': 'Insider Threat'},   # Diamond (Hidden)
    'Dynamic Def':    {'color': '#263238', 'marker': 'o', 'label': 'Dynamic Defense'},  # Circle (Adaptive)
    'Static Def':     {'color': '#90A4AE', 'marker': '^', 'label': 'Static Defense'}    # Triangle (Baseline)
}

# Matplotlib setup to follow Academic/Tufte standards
# - Serif fonts (Times New Roman) looks professional in LaTeX
# - 'spines.top/right = False' removes the "box" (chartjunk)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,        # High res for printing
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False, 
    'axes.spines.right': False,
    'lines.linewidth': 1.5
})

CKC_ORDER = ["Reconnaissance", "Weaponization", "Delivery", "Exploitation",
             "Installation", "Command & Control", "Actions on Objectives"]

def get_academic_label(raw_name):
    """
    Helper: Maps experiment raw names to paper taxonomy.
    """
    if 'Hardened' in raw_name: return 'Hardened Def'
    if 'Insider' in raw_name:  return 'Insider Threat'
    if 'Adaptive' in raw_name: return 'Dynamic Def' 
    return 'Static Def'                             

# -----------------------
# PLOTTING FUNCTIONS
# -----------------------

def plot_survival(conn):
    """
    Fig 1: Attacker Survival Rate (Line Plot).
    
    Visualization:
        X-Axis: CKC Stages (0-6)
        Y-Axis: % Remaining
        Highlight: Shaded 'Choke Point' at Exploitation.
    """
    print("Generating Fig 1: Survival...")
    
    # Query MAX step reached per episode to get survival depth
    df = pd.read_sql("""
        SELECT e.name, ep.run_idx, MAX(s.step_idx) as max_step
        FROM Steps s
        JOIN Episodes ep ON s.episode_id = ep.episode_id
        JOIN Experiments e ON ep.experiment_id = e.experiment_id
        WHERE e.name != 'Baseline_Rational'
        GROUP BY e.name, ep.run_idx
    """, conn)
    if df.empty: return

    df['Label'] = df['name'].apply(get_academic_label)
    
    # Calc % survivors at each stage
    plot_data = []
    for label in df['Label'].unique():
        sub = df[df['Label'] == label]
        total = len(sub)
        for i, stage in enumerate(CKC_ORDER):
            # How many survived at least to stage 'i'?
            count = len(sub[sub['max_step'] >= i])
            pct = (count / total) * 100
            plot_data.append({'Experiment': label, 'Stage': stage, 'Survival': pct})
            
    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot Lines
    for label in df_plot['Experiment'].unique():
        subset = df_plot[df_plot['Experiment'] == label]
        style = STYLE_MAP[label]
        ax.plot(subset['Stage'], subset['Survival'], 
                color=style['color'], marker=style['marker'], 
                label=style['label'], markersize=6, 
                markeredgecolor='white', markeredgewidth=0.5)

    ax.set_title('Attacker Survival Rate by Kill Chain Stage')
    ax.set_ylabel('Survival Probability (%)')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=25, ha='right')
    
    # Visual Annotation: The "Choke Point"
    # Shading the Exploitation phase draws the eye immediately.
    ax.axvspan(2.5, 3.5, color='#ECEFF1', alpha=0.5, zorder=0)
    ax.text(3, 10, "Exploitation\nChoke Point", ha='center', fontsize=8, color='#546E7A')

    ax.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "FIG1_Survival.png"))
    plt.close()

def plot_cost_trade_off(conn):
    """
    Fig 2: Cost vs Impact (Scatter).
    
    Visualization:
        X-Axis: Cost (Log Scale) - Money spent
        Y-Axis: Utility - Damage taken
        Goal: Bottom-Left (Low Cost/Low Impact)
    """
    print("Generating Fig 2: Cost and Resilience Trade-Off...")
    df = pd.read_sql("""
        SELECT e.name, 
               AVG(ep.total_defense_cost) as Cost,
               AVG(ep.total_attacker_utility) as Impact,
               AVG(ep.success) * 100 as Success_Rate
        FROM Episodes ep
        JOIN Experiments e ON ep.experiment_id = e.experiment_id
        WHERE e.name != 'Baseline_Rational'
        GROUP BY e.name
    """, conn)
    if df.empty: return

    df['Label'] = df['name'].apply(get_academic_label)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, row in df.iterrows():
        lbl = row['Label']
        style = STYLE_MAP[lbl]
        
        ax.scatter(row['Cost'], row['Impact'], 
                   c=style['color'], marker=style['marker'], s=100, 
                   edgecolor='black', linewidth=0.5, zorder=3)
        
        # Smart text offset so labels don't overlap dots
        offset = (0, 12) if 'Hardened' in lbl else (0, -18)
        va = 'bottom' if 'Hardened' in lbl else 'top'
        if 'Insider' in lbl: offset = (0, -22)

        ax.annotate(f"{style['label']}\n({row['Success_Rate']:.0f}% Succ)", 
                    (row['Cost'], row['Impact']), 
                    xytext=offset, textcoords='offset points',
                    ha='center', va=va, fontsize=9, color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Arrow pointing to "Optimal" zone
    min_cost, min_imp = df['Cost'].min(), df['Impact'].min()
    ax.annotate('Optimal Zone\n(Low Cost / Low Impact)', 
                xy=(min_cost, min_imp), xytext=(min_cost, min_imp * 0.5),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2),
                ha='center', fontsize=9, color='#2E7D32', fontweight='bold')

    # Log Scale needed because costs vary exponentially
    ax.set_xscale('log')
    ax.set_title('Cost and Resilience Trade-Off (Cost vs. Impact)')
    ax.set_xlabel('Defense Cost [MU] (Log Scale)')
    ax.set_ylabel('Attacker Utility [MU]')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "FIG2_Cost_Trade_Off.png"))
    plt.close()

def plot_signal_to_noise(conn):
    """
    Fig 3: Operational Load (Noise vs Signal).
    
    Visualization:
        X-Axis: False Positives (Noise)
        Y-Axis: Blocking Rate (Signal)
        Zone: Alert Fatigue is bottom-right.
    """
    print("Generating Fig 3: Signal to Noise...")
    df = pd.read_sql("""
        SELECT e.name, 
               AVG(ep.total_falsepos_actions) as False_Alarms,
               (COUNT(CASE WHEN ep.success = 0 THEN 1 END) * 1.0 / COUNT(*)) * 100 as Block_Rate
        FROM Episodes ep
        JOIN Experiments e ON ep.experiment_id = e.experiment_id
        WHERE e.name != 'Baseline_Rational'
        GROUP BY e.name
    """, conn)
    if df.empty: return
    
    df['Label'] = df['name'].apply(get_academic_label)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    for i, row in df.iterrows():
        lbl = row['Label']
        style = STYLE_MAP[lbl]
        ax.scatter(row['False_Alarms'], row['Block_Rate'], 
                   c=style['color'], marker=style['marker'], s=100, 
                   edgecolor='black', linewidth=0.5, zorder=3)
        
        offset = (0, 10) if 'Hardened' in lbl else (0, -15)
        ax.annotate(style['label'], (row['False_Alarms'], row['Block_Rate']), 
                    xytext=offset, textcoords='offset points',
                    ha='center', fontsize=9, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Draw Quadrant lines to show "Good" vs "Noisy"
    mid_x = (df['False_Alarms'].max() + df['False_Alarms'].min()) / 2
    mid_y = (df['Block_Rate'].max() + df['Block_Rate'].min()) / 2
    ax.axhline(mid_y, color='grey', linestyle=':', linewidth=1)
    ax.axvline(mid_x, color='grey', linestyle=':', linewidth=1)

    ax.text(0.98, 0.98, "High Load Area", transform=ax.transAxes, 
            ha='right', va='top', fontsize=8, color='grey', style='italic')
    ax.text(0.98, 0.02, "Alert Fatigue Zone", transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=8, color='grey', style='italic')

    ax.set_title('Operational Load (Noise vs. Block Rate)')
    ax.set_xlabel('False Positives per Attack (Noise)')
    ax.set_ylabel('Blocking Rate (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "FIG3_Signal_Noise.png"))
    plt.close()

def plot_comparative_bars(conn):
    """
    Fig 4: Bar Charts (Cost & MTTD).
    
    Visualization:
        1. Total Defense Cost (Log Scale)
        2. Mean Time To Detect (MTTD)
    """
    print("Generating Fig 4: Metrics...")
    df = pd.read_sql("""
        SELECT e.name, 
               AVG(ep.total_defense_cost) as Cost,
               AVG(ep.mean_time_to_detect) as MTTD
        FROM Episodes ep JOIN Experiments e ON ep.experiment_id = e.experiment_id
        GROUP BY e.name
    """, conn)
    if df.empty: return
    
    df['Label'] = df['name'].apply(get_academic_label)
    
    # Force sort order: Static -> Dynamic -> Hardened -> Insider
    sort_order = ['Static Def', 'Dynamic Def', 'Hardened Def', 'Insider Threat']
    df['Label'] = pd.Categorical(df['Label'], categories=sort_order, ordered=True)
    df = df.sort_values('Label')

    labels = df['Label'].tolist()
    costs = df['Cost'].tolist()
    mttds = df['MTTD'].tolist()
    colors = [STYLE_MAP[l]['color'] for l in labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    
    # Subplot 1: Cost
    ax1.bar(labels, costs, color=colors, alpha=0.9, width=0.6, edgecolor='black', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_title('Total Defense Cost')
    ax1.set_ylabel('Monetary Units (Log)')
    ax1.tick_params(axis='x', rotation=20)

    # Subplot 2: MTTD
    ax2.bar(labels, mttds, color=colors, alpha=0.9, width=0.6, edgecolor='black', linewidth=0.5)
    ax2.set_title('Mean Time to Detect (MTTD)')
    ax2.set_ylabel('Steps')
    ax2.tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "FIG4_Comparative_Metrics.png"))
    plt.close()

# -----------------------
# MAIN
# -----------------------
def run_final_visuals():
    """Main Driver: Connects DB -> runs plots."""
    if not os.path.exists(DB_FILE): return
    conn = sqlite3.connect(DB_FILE)
    try:
        plot_survival(conn)
        plot_cost_trade_off(conn)
        plot_signal_to_noise(conn)
        plot_comparative_bars(conn)
        print("\nSUCCESS. Academic Figures generated in './Phase2_Outputs/'.")
    finally:
        conn.close()

if __name__ == "__main__":
    run_final_visuals()