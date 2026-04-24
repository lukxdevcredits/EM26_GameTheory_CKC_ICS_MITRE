#!/usr/bin/env python3
"""
Phase-2 Analysis
Author: LBD
Date: 2025
Description: Generates plots for Phase 2 results.
             - Reads CSV output
             - Visualizes attacker learning & chosen strategies
             - Exports .png for quick review
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings (keeps the terminal clean during batch runs)
warnings.simplefilter(action='ignore', category=FutureWarning)

# -----------------------
# CONF, CONSTANTS
# -----------------------
# specific order for axes so charts follow the timeline
CKC_STAGES = ["Reconnaissance", "Weaponization", "Delivery", "Exploitation", 
              "Installation", "Command & Control", "Actions on Objectives"]

STRATEGY_ORDER = ['Stealthy', 'Standard', 'Aggressive']

OUTPUT_DIR = "./Phase2_Outputs"

# -----------------------
# STYLE
# -----------------------
# color mapping for readability
STRAT_COLORS = {
    'Stealthy':   '#558B2F',  # Green (Safe/Quiet)
    'Standard':   '#546E7A',  # Grey (Neutral)
    'Aggressive': '#BF360C'   # Red (Danger/Loud)
}

OUTCOME_COLORS = {
    'Blocked': '#455A64', # Dark Grey (Defended)
    'Success': '#C5A059'  # Gold (Attacker Won)
}

# Matplotlib defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False
})


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def get_strategy_from_name(name):
    """
    Parses complex technique names into 3 simple buckets.
    
    Logic:
        Naive keyword matching. Used because raw logs don't always 
        store the high-level category explicitly.
    """
    name = str(name).lower()
    if any(x in name for x in ['sniffing', 'masquerading', 'zero-day', 'covert', 'quiet']):
        return 'Stealthy'
    elif any(x in name for x in ['high-volume', 'worm', 'backdoor', 'loss of safety']):
        return 'Aggressive'
    else:
        return 'Standard'
    
# -----------------------
# MAIN ANALYSIS
# -----------------------
def run_analysis():
    """
    Loads data -> generates plots.
    """
    
    # 1. Load data
    try:
        df_episodes = pd.read_csv(os.path.join(OUTPUT_DIR, "episodes.csv"))
        df_steps = pd.read_csv(os.path.join(OUTPUT_DIR, "steps.csv"))
    except FileNotFoundError:
        print("Error: Files not found. Please run sim_phase2.py first.")
        return

    print(f"Loaded {len(df_episodes)} episodes.")

    # 2. Consoles stats (sanity check)
    success_rate = (df_episodes['success'].sum() / len(df_episodes)) * 100
    avg_steps = df_steps.groupby('run_idx').size().mean()
    avg_reward = df_episodes['total_attacker_utility'].mean()

    print("\n--- PHASE 2 RESULTS ---")
    print(f"Success Rate:         {success_rate:.1f}%")
    print(f"Avg Steps Survived:   {avg_steps:.1f} Steps")
    print(f"Avg Attacker Reward:  {avg_reward:.0f} Points")
    print("-----------------------\n")

    # sort by Categorical Ordering
    df_steps['ck_stage'] = pd.Categorical(df_steps['ck_stage'], categories=CKC_STAGES, ordered=True)
    df_steps = df_steps.sort_values('ck_stage')
    df_steps['Strategy'] = df_steps['technique_name'].apply(get_strategy_from_name)

    # Plot 1: Belief (Learning Curve)
    #  Show if the attacker is getting scared (Belief -> 1.0) or bold (Belief -> 0.0)
    print("Plotting Fig 1: Belief...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Avg belief per stage across all runs
    belief_data = df_steps.groupby('ck_stage')['attacker_belief_high_cap'].mean().reset_index()

    ax.plot(belief_data['ck_stage'], belief_data['attacker_belief_high_cap'], 
            marker='o', linewidth=3, color='#263238', label='Attacker Confidence')
    
    # Reference line: 0.5 is a coin toss (Start)
    ax.axhline(y=0.5, color='grey', linestyle='--', label='Start (50/50 Guess)')

    ax.set_title("How the Attacker's Belief Changes")
    ax.set_ylabel("Belief that Defender is Strong (0-1)")
    ax.set_xlabel("Attack Stage")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=20, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_belief_simple.png"))
    plt.close()

    # Plot 2: Strategy Choice (stacked bar)
    #  Visualize the shift from Stealth (green) to Aggression (red) over time.
    print("Plotting Fig 2: Strategy...")
    fig, ax = plt.subplots(figsize=(9, 5))

    # Pivot Data: Rows=Stage, Cols=Strategy, Values=Count
    strat_counts = df_steps.groupby(['ck_stage', 'Strategy'], observed=False).size().unstack(fill_value=0)
    
    # Normalize to 100% to see distribution
    strat_pct = strat_counts.div(strat_counts.sum(axis=1), axis=0) * 100
    
    # FIX: Use reindex to prevent crash if a strategy (e.g. 'Aggressive') never happened
    strat_pct = strat_pct.reindex(columns=STRATEGY_ORDER, fill_value=0)
    colors = [STRAT_COLORS[s] for s in STRATEGY_ORDER]

    strat_pct.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.8)

    ax.set_title("Attacker Strategy Used at Each Stage")
    ax.set_ylabel("Percentage of Time Used (%)")
    ax.set_xlabel("Attack Stage")
    plt.xticks(rotation=20, ha='right')
    ax.legend(title="Strategy Type", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_strategy_simple.png"))
    plt.close()

    # Plot 3: Rewards (Utility)
    #  show where the "Big Wins" are for the attacker.
    print("Plotting Fig 3: Rewards...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    reward_data = df_steps.groupby('ck_stage')['discounted_payoff_step'].mean().reset_index()

    ax.bar(reward_data['ck_stage'], reward_data['discounted_payoff_step'], 
           color='#C5A059', edgecolor='black', alpha=0.8)

    ax.set_title("Average Reward for the Attacker per Stage")
    ax.set_ylabel("Reward Points (Utility)")
    ax.set_xlabel("Attack Stage")
    plt.xticks(rotation=20, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_rewards_simple.png"))
    plt.close()

    # Plot 4: Outcomes (The Funnel)
    #  See where attacks actually die.
    print("Plotting Fig 4: Outcomes...")
    
    # Find the FINAL stage reached for every run
    last_stages = df_steps.groupby('run_idx').last()['ck_stage'].reset_index()
    last_stages.rename(columns={'ck_stage': 'End_Stage'}, inplace=True)
    
    # Join with episode result (Did they win or get blocked?)
    data = pd.merge(last_stages, df_episodes[['run_idx', 'success']], on='run_idx')
    data['Result'] = data['success'].apply(lambda x: 'Success' if x else 'Blocked')
    
    # Count frequency per stage
    counts = data.groupby(['End_Stage', 'Result'], observed=False).size().unstack(fill_value=0)
    # Fill missing stages with 0 so x-axis is complete
    counts = counts.reindex(CKC_STAGES, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    counts[['Blocked', 'Success']].plot(kind='bar', stacked=True, 
                                        color=[OUTCOME_COLORS['Blocked'], OUTCOME_COLORS['Success']], 
                                        ax=ax, width=0.8, edgecolor='black')

    ax.set_title("Where Attacks Stop")
    ax.set_ylabel("Number of Attempts")
    ax.set_xlabel("Stage where Attack Ended")
    plt.xticks(rotation=20, ha='right')
    ax.legend(title="Outcome")

    # Avg stopping point line
    ax.axvline(x=avg_steps - 1, color='red', linestyle=':', linewidth=2, label="Avg Stopping Point")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_outcomes_simple.png"))
    plt.close()

    print("Done! Simple plots saved to ./Phase2_Outputs/")

if __name__ == "__main__":
    run_analysis()