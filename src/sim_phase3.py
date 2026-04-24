#!/usr/bin/env python3
"""
Phase 3 - DATABASE SETUP & IMPORT
Author: LBD
Date: 2025
Description: Initializes SQLite DB (phase2.db) 
             - Creates Game-Theoretic schema
             - Imports Phase 2 CSV results for baseline comparison
"""

import sqlite3
import pandas as pd
import json
import os

# -----------------------
# CONFIG & PATHS
# -----------------------
INPUT_DIR = './Phase2_Outputs'
OUTPUT_DB = 'phase2.db'
CONFIG_JSON = 'config_phase2.json'

# Abs paths for source CSVs
EPISODES_CSV = os.path.join(INPUT_DIR, 'episodes.csv')
STEPS_CSV = os.path.join(INPUT_DIR, 'steps.csv')

def init_db():
    """
    Init the SQLite DB structure.

    Params:
        None
    
    Returns:
        sqlite3.Connection: Active connection
    
    Logic:
        1. Deletes old DB to start clean (no schema conflicts)
        2. Creates 3 normalized tables (Experiments, Episodes, Steps)
    """
    
    # Clean State: Delete old DB if exists
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)
        print(f"Removed old {OUTPUT_DB} to ensure clean schema.")

    # Connect to DB (creates file if missing)
    conn = sqlite3.connect(OUTPUT_DB)
    cur = conn.cursor()

    # Table 1: Experiments
    # Stores unique names and settings (e.g. Monitor Level)
    cur.execute("""
    CREATE TABLE Experiments (
        experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        monitor_level TEXT,
        description TEXT,
        N_episodes INTEGER
    )""")

    # Table 2: Episodes
    # Summary of single Monte-Carlo run
    # 'total_attacker_utility' tracks adversary payoff
    cur.execute("""
    CREATE TABLE Episodes (
        episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        run_idx INTEGER,
        success BOOLEAN,
        total_downtime REAL,
        total_damage REAL,
        total_defense_cost REAL,
        total_falsepos_actions INTEGER,
        mean_time_to_detect REAL,
        total_attacker_utility REAL, 
        FOREIGN KEY (experiment_id) REFERENCES Experiments(experiment_id)
    )""")

    # Table 3: Steps
    # Details for each CKC stage (0-7)
    # 'nash_prob_chosen' validates strategy selection
    cur.execute("""
    CREATE TABLE Steps (
        step_id INTEGER PRIMARY KEY AUTOINCREMENT,
        episode_id INTEGER,
        step_idx INTEGER,
        ck_stage TEXT,
        technique_id TEXT,
        technique_name TEXT,
        action_success BOOLEAN,
        detected BOOLEAN,
        detect_prob REAL,
        detection_delay INTEGER,
        impact_minutes REAL,
        attacker_belief_high_cap REAL,
        nash_prob_chosen REAL,
        discounted_payoff_step REAL,
        FOREIGN KEY (episode_id) REFERENCES Episodes(episode_id)
    )""")

    # Commit schema
    conn.commit()
    print(f"Database initialized: {OUTPUT_DB}")
    return conn

def import_phase2_data(conn):
    """
    Imports historical simulation data into DB.

    Params:
        conn (sqlite3.Connection): Active DB connection
    
    Logic:
        - Validates CSVs exist
        - Maps 'Monitor Levels' from JSON config
        - Bulk inserts Episodes and Steps
        - Joins CSV 'run_idx' to SQL 'episode_id' for linkage
    """
    
    # Validation: Check input files exist
    if not (os.path.exists(EPISODES_CSV) and os.path.exists(STEPS_CSV)):
        print(f"Phase 2 CSVs not found in {INPUT_DIR}. Skipping Baseline import.")
        return

    print("Importing Phase 2 (Baseline) CSV data...")
    
    # Load raw CSVs
    df_ep = pd.read_csv(EPISODES_CSV)
    df_st = pd.read_csv(STEPS_CSV)
    
    cur = conn.cursor()

    # Load Config for metadata (defaults to empty if missing)
    try:
        with open(CONFIG_JSON, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {'experiments': []}

    # Find unique experiments
    unique_experiments = df_ep['experiment'].unique()
    
    for exp_name in unique_experiments:
        print(f"Processing Experiment: {exp_name}...")
        
        # Metadata Lookup: Get settings from config
        exp_conf = next((e for e in config['experiments'] if e['name'] == exp_name), {})
        monitor_level = exp_conf.get('monitor_level', 'Baseline')
        
        # Filter data for current experiment
        subset_ep = df_ep[df_ep['experiment'] == exp_name].copy()
        n_episodes = len(subset_ep)
        desc = "Phase 2 Baseline Run"

        # 1. Insert Experiment Record
        # Ignore duplicates to avoid errors
        cur.execute("""
            INSERT OR IGNORE INTO Experiments (name, monitor_level, description, N_episodes)
            VALUES (?, ?, ?, ?)
        """, (exp_name, monitor_level, desc, n_episodes))
        
        # Get generated Experiment ID
        cur.execute("SELECT experiment_id FROM Experiments WHERE name=?", (exp_name,))
        experiment_id = cur.fetchone()[0]

        # 2. Bulk Insert Episodes
        episode_records = []
        for _, row in subset_ep.iterrows():
            episode_records.append((
                experiment_id, row['run_idx'], row['success'], row['total_downtime'],
                row['total_damage'], row['total_defense_cost'], row['total_falsepos_actions'],
                row['mean_time_to_detect'], row['total_attacker_utility']
            ))

        cur.executemany("""
        INSERT INTO Episodes (experiment_id, run_idx, success, total_downtime, total_damage,
                              total_defense_cost, total_falsepos_actions, mean_time_to_detect,
                              total_attacker_utility)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, episode_records)
        conn.commit()

        # 3. Bulk Insert Steps
        # Logic: Link Steps to parent Episode via 'run_idx' join
        df_ep_ids = pd.read_sql("SELECT episode_id, run_idx FROM Episodes WHERE experiment_id=?", 
                                conn, params=(experiment_id,))
        
        subset_st = df_st[df_st['experiment'] == exp_name].copy()
        
        # Merge Step data with Episode IDs
        df_merge = pd.merge(subset_st, df_ep_ids, on='run_idx', how='inner')

        step_records = []
        for _, row in df_merge.iterrows():
            step_records.append((
                row['episode_id'], row['step_idx'], row['ck_stage'], row['technique_id'],
                row['technique_name'], row['action_success'], row['detected'],
                row['detect_prob'], row['detection_delay'], row['impact_minutes'],
                row['attacker_belief_high_cap'], row['nash_prob_chosen'], row['discounted_payoff_step']
            ))

        cur.executemany("""
        INSERT INTO Steps (episode_id, step_idx, ck_stage, technique_id, technique_name,
                           action_success, detected, detect_prob, detection_delay, impact_minutes,
                           attacker_belief_high_cap, nash_prob_chosen, discounted_payoff_step)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, step_records)
        conn.commit()
        
        print(f"  -> Imported {len(episode_records)} episodes and {len(step_records)} steps.")

if __name__ == "__main__":
    # Main Execution
    conn = init_db()            # Init DB
    import_phase2_data(conn)    # Import Data
    conn.close()                # Close
    
    # Verification: Confirm import
    print("\n=== Verification Queries ===")
    conn = sqlite3.connect(OUTPUT_DB)
    df_ver = pd.read_sql("SELECT name, N_episodes FROM Experiments", conn)
    print(df_ver)
    conn.close()