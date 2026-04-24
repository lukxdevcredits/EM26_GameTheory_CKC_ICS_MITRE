#!/usr/bin/env python3
"""
Phase 4 - ICS GAME-THEORETIC SOLVER
Author: LBD
Date: 2025
Description: Executes the Game-Theoretic simulation.
             - Solves 3x3 Interaction Matrices (Attacker vs Defender).
             - Implements Bayesian Belief Updating for Attacker.
             - Calculates 'Hardened' and 'Insider' scenarios.
"""

import json, sqlite3, os
import numpy as np
import pandas as pd
from math import exp
import nashpy as nash 
import warnings

# Silence degeneracy warnings (keeps console clean)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nashpy")

# -----------------------
# CONFIG & PATHS
# -----------------------
INPUT_DIR = '.' 
DB_FILE = 'phase2.db' 
OUTPUT_DIR = './Phase2_Outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_JSON = os.path.join(INPUT_DIR, 'attack_techniques.json')
PAYOFF_JSON = os.path.join(INPUT_DIR, 'payoff_model.json')
CONFIG_JSON = os.path.join(INPUT_DIR, 'config_phase2.json')

np.random.seed(12345)

# Load system models
with open(ATTACK_JSON,'r') as f: attack_techniques = json.load(f)
with open(PAYOFF_JSON,'r') as f: payoff_model = json.load(f)
with open(CONFIG_JSON,'r') as f: config = json.load(f)

# Unpack
P_CONF = payoff_model['system_parameters']
B_CONF = payoff_model['bayesian_parameters']
PO_CONF = payoff_model['payoff_parameters']
D_CONF = payoff_model['detection_parameters']
SENS_CONF = payoff_model['strategic_sensitivity']
ALPHA_FP_CONF = payoff_model['strategic_false_positive']

# group techniques by Stage -> Strategy for fast lookup
techs_by_stage_strategy = {}
for group in attack_techniques['ics_attack_tactics']:
    stage = group['ckc_stage']
    if stage not in techs_by_stage_strategy:
        techs_by_stage_strategy[stage] = {'Stealthy': [], 'Standard': [], 'Aggressive': []}
    for t in group['techniques']:
        techs_by_stage_strategy[stage][t['strategy_class']].append(t)
            
CKC_ORDER = ["Reconnaissance", "Weaponization", "Delivery", "Exploitation",
             "Installation", "Command & Control", "Actions on Objectives"]
DEFENDER_ACTIONS = ['low', 'med', 'high']

# -----------------------
# CORE LOGIC
# -----------------------

def calculate_step_payoffs(technique, monitor_level, stage, step_idx, is_insider, is_hardened):
    """
    Computes utility (payoff) for a single interaction.
    
    Params:
        technique (dict): The MITRE technique data
        monitor_level (str): 'low', 'med', or 'high'
        stage (str): Current CKC stage name
        step_idx (int): 0-6 index
        is_insider (bool): Apply 0.2x detection mod?
        is_hardened (bool): Apply 1.5x sensitivity mod?

        Attacker Utility = Gain - Cost - Penalty(if caught)
        Defender Loss    = Defense Cost + Damage(if failed)
    """
    strategy_class = technique['strategy_class']
    
    # 1. Base Variables
    base_gain = PO_CONF.get(f'attacker_{strategy_class.lower()}_gain', 0)
    attacker_cost = technique['attacker_cost']
    success_prob = technique['base_success_prob']
    
    # 2. Resilience & Detection
    # As time passes (step increases), Defender gets smarter (+5% per step)
    resilience_factor = 1.0 + (0.05 * step_idx)
    
    # Hardened defense is strictly 1.5x more sensitive
    sens_multiplier = 1.5 if is_hardened else 1.0
    current_sens = SENS_CONF[monitor_level] * sens_multiplier
    
    # Final detect chance = Base Tech * Sensitivity * Time Resilience
    p_detect_base = technique['base_detectability_lambda'] * current_sens
    p_detect = min(0.99, p_detect_base * resilience_factor)
    
    # Insider Logic: They have credentials, so 80% harder to detect
    if is_insider and strategy_class == 'Standard':
        p_detect = p_detect * 0.2 
        
    expected_penalty = 0
    if p_detect > 0:
        penalty_key = f'penalty_{strategy_class.lower()}_detection'
        detection_penalty = PO_CONF.get(penalty_key, PO_CONF.get('penalty_standard_detection', 0))
        # Expected Penalty = Chance of getting caught * The Penalty
        expected_penalty = p_detect * detection_penalty
        
    # 3. Asset Value & Dynamic Bias
    FUTURE_GAIN_BASE = P_CONF['V_objective_map'].get(stage, 10000)
    
    # "Dynamic Bias": Psych pressure. 
    # Rewards 'Stealth' early on, and 'Aggression' near the end.
    stage_idx_val = CKC_ORDER.index(stage)
    normalized_idx = stage_idx_val / (len(CKC_ORDER) - 1) 
    adjustment_factor = 0
    if strategy_class == 'Stealthy': adjustment_factor = 150 * (1 - normalized_idx) 
    elif strategy_class == 'Aggressive': adjustment_factor = 150 * normalized_idx
    elif strategy_class == 'Standard': adjustment_factor = 75 
        
    # Calc Future Utility: (Prob Success) * (Value)
    expected_future_utility = (1 - p_detect) * success_prob * (FUTURE_GAIN_BASE + adjustment_factor)
    
    # Total Payoff
    step_payoff = base_gain - attacker_cost + expected_penalty
    attacker_utility = expected_future_utility + step_payoff
    
    # 4. Defender Loss
    # Sum of running the tools (Cost) + leakage (Unstopped Damage)
    cost_of_defense = payoff_model.get(f'C_def_{monitor_level}', 0) 
    damage_if_not_detected = (1 - p_detect) * FUTURE_GAIN_BASE
    defender_loss = cost_of_defense + damage_if_not_detected
    
    return attacker_utility, defender_loss

def create_payoff_matrix(ckc_stage, step_idx, is_insider, is_hardened):
    """
    Builds the 3x3 Payoff Matrix.
    
    Params:
        ckc_stage (str): Current stage name
        step_idx (int): Current stage index (0-6)
        is_insider, is_hardened (bool): Scenario flags

    Returns:
        (A, B): Tuple of numpy arrays (Attacker, -Defender)
    """
    ATTACKER_STRATEGIES = ['Stealthy', 'Standard', 'Aggressive']
    A = np.zeros((3, 3)) 
    B = np.zeros((3, 3)) 
    
    # Pick the best technique for each strategy (Optimal Move)
    representative_techs = {}
    for strategy in ATTACKER_STRATEGIES:
        techs = techs_by_stage_strategy.get(ckc_stage, {}).get(strategy, [])
        if techs:
            representative_techs[strategy] = max(techs, key=lambda t: t['base_success_prob'])
        else:
             representative_techs[strategy] = {'strategy_class': strategy, 'attacker_cost': 0, 'base_success_prob': 0.0, 'base_detectability_lambda': 0.0, 'id': 'T000', 'name': 'No-Op'}
             
    # Fill matrix
    for i, att_strat in enumerate(ATTACKER_STRATEGIES):
        tech = representative_techs[att_strat]
        for j, def_level in enumerate(DEFENDER_ACTIONS):
            att_util, def_loss = calculate_step_payoffs(tech, def_level, ckc_stage, step_idx, is_insider, is_hardened)
            A[i, j] = att_util
            B[i, j] = def_loss

    # Add small noise
    # Solvers crash on exact ties, this breaks them (+/- 0.000000001).
    epsilon = 1e-9
    A += np.random.uniform(-epsilon, epsilon, A.shape)
    B += np.random.uniform(-epsilon, epsilon, B.shape)
            
    return A, -B

def solve_nash_equilibrium(A, B):
    """
    Finds Mixed-Strategy Nash Equilibrium (MSNE).
    
    Params:
        A (np.array): Attacker matrix
        B (np.array): Defender matrix
        
    Returns:
        (p_row, p_col): Optimal probabilities for Attacker/Defender
    """
    game = nash.Game(A, B)
    equilibria = list(game.support_enumeration())
    
    # Fallback: If solver breaks (rare edge case), default to random.
    if not equilibria:
        return np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3])
    return equilibria[0] 

def select_nash_strategy(nash_probs):
    # Rolls the dice based on Nash weights
    return np.random.choice([0, 1, 2], p=nash_probs)

def bayesian_update(current_belief, detected):
    """
    Updates Attacker Belief (Theta) using Bayes Rule.
    
    Params:
        current_belief (float): Prob of High Cap Defender (0.0-1.0)
        detected (bool): Did the last attack get caught?

    Equation: P(Strong | Observation)
    """
    if detected:
        # If caught: likely a strong defender
        P_E_H = B_CONF['likelihood_block_high_cap'] # e.g., 0.9
        P_E_L = B_CONF['likelihood_block_low_cap']  # e.g., 0.1
    else:
        # If evasion: invert likelihoods
        P_E_H = 1 - B_CONF['likelihood_block_high_cap']
        P_E_L = 1 - B_CONF['likelihood_block_low_cap']
        
    P_H = current_belief 
    P_L = 1 - P_H        
    
    # Denominator: Total Prob of Evidence
    P_E = (P_E_H * P_H) + (P_E_L * P_L)
    
    if P_E == 0: return current_belief 
    new_belief = (P_E_H * P_H) / P_E
    
    # Clip to 1-99% to prevent math errors on future updates
    return np.clip(new_belief, 0.01, 0.99)

def calculate_tcd(base_cost, total_damage_minutes, total_falsepos_actions):
    """
    Calc Total Cost of Defense (TCD).
    
    Params:
        base_cost (float): Sum of tool costs per step
        total_damage_minutes (float): Cumulative impact
        total_falsepos_actions (int): Count of false alarms
        
    TCD = Tool Cost + Damage + Fatigue
    """
    C_def_base_cumulative = base_cost 
    T_max = P_CONF['T_max']
    
    # Exp Penalty: 10 mins damage is annoying, 100 mins is fatal.
    loss_penalty_factor = exp(total_damage_minutes / T_max) - 1 
    loss_value = (total_damage_minutes * P_CONF['C_downtime_per_min']) * loss_penalty_factor
    
    fp_cost = total_falsepos_actions * P_CONF['C_falsepos_maint']
    return C_def_base_cumulative + loss_value + fp_cost

# -----------------------
# MAIN SIMULATION
# -----------------------
all_episode_rows = []
all_step_records = []

for solver_experiment in config['experiments']:
    print(f"Running Experiment: {solver_experiment['name']}...")
    is_insider = "Insider" in solver_experiment['name']
    is_hardened = "Hardened" in solver_experiment['name']
    
    for run_idx in range(solver_experiment['N_episodes']):
        # Init Episode
        episode_success = False
        episode_damage_minutes = 0
        episode_false_positives = 0
        attacker_belief_high_cap = B_CONF['initial_belief_high_cap'] 
        current_utility = 0
        discount_factor = P_CONF['discount_factor']
        episode_defense_cost_base = 0 

        # Loop through Stages (0-6)
        for step_idx, stage in enumerate(CKC_ORDER):
            
            # 1. Game Theory: Solve Matrix
            A_matrix, B_matrix = create_payoff_matrix(stage, step_idx, is_insider, is_hardened)
            attacker_nash_probs, defender_nash_probs = solve_nash_equilibrium(A_matrix, B_matrix)
            
            # 2. Select Strategies
            attacker_strat_idx = select_nash_strategy(attacker_nash_probs)
            attacker_strat_name = ['Stealthy', 'Standard', 'Aggressive'][attacker_strat_idx]
            
            defender_strat_idx = select_nash_strategy(defender_nash_probs)
            monitor_level = DEFENDER_ACTIONS[defender_strat_idx]
            
            # Track Costs
            def_base_cost_step = payoff_model.get(f'C_def_{monitor_level}', 0) 
            episode_defense_cost_base += def_base_cost_step 
            
            # 3. Execution
            available_techs = techs_by_stage_strategy.get(stage, {}).get(attacker_strat_name, [])
            if not available_techs: continue
            technique = max(available_techs, key=lambda t: t['base_success_prob'])
            
            action_success = np.random.rand() < technique['base_success_prob']
            
            # Detection Check
            resilience_factor = 1.0 + (0.05 * step_idx)
            sens_multiplier = 1.5 if is_hardened else 1.0
            p_detect_base = technique['base_detectability_lambda'] * SENS_CONF[monitor_level] * sens_multiplier
            p_detect = min(0.99, p_detect_base * resilience_factor)
            
            if is_insider and attacker_strat_name == 'Standard':
                p_detect = p_detect * 0.2
                
            detected = np.random.rand() < p_detect
            detection_delay = 1 if detected else 0
            
            # False Positive: Did we alert even though we missed them?
            if not detected and np.random.rand() < ALPHA_FP_CONF[monitor_level]:
                episode_false_positives += 1

            impact_minutes = 0
            if action_success and not detected and stage == "Actions on Objectives":
                 impact_minutes = technique['impact_magnitude']
                 episode_damage_minutes += impact_minutes
                 episode_success = True
                 
            # 4. Updates & Learning
            step_attacker_utility = A_matrix[attacker_strat_idx, defender_strat_idx]
            discounted_payoff_step = step_attacker_utility * (discount_factor ** step_idx)
            current_utility += discounted_payoff_step
            
            attacker_belief_high_cap = bayesian_update(attacker_belief_high_cap, detected)
            
            step_record = (
                solver_experiment['name'], run_idx, step_idx, stage, technique['id'], technique['name'],
                action_success, detected, p_detect, detection_delay, impact_minutes,
                attacker_belief_high_cap, attacker_nash_probs[attacker_strat_idx], discounted_payoff_step
            )
            all_step_records.append(step_record)
            
            # End Condition: Attacker Wins (Obj) or Defender Wins (Detected)
            if episode_success or detected:
                break
                
        # Finalize
        total_defense_cost = calculate_tcd(episode_defense_cost_base, episode_damage_minutes, episode_false_positives)
        mttd = step_idx + 1 if detected else None
        
        episode_row = (
            solver_experiment['name'], config.get('seed',12345), run_idx, episode_success,
            episode_damage_minutes, episode_damage_minutes, total_defense_cost,
            episode_false_positives, mttd, current_utility 
        )
        all_episode_rows.append(episode_row)

# -----------------------
# DATA PERSISTENCE
# -----------------------
columns_episodes = ['experiment','seed','run_idx','success','total_downtime','total_damage',
                    'total_defense_cost','total_falsepos_actions','mean_time_to_detect',
                    'total_attacker_utility']
columns_steps = ['experiment','run_idx','step_idx','ck_stage','technique_id','technique_name',
                 'action_success','detected','detect_prob','detection_delay','impact_minutes',
                 'attacker_belief_high_cap', 'nash_prob_chosen', 'discounted_payoff_step']

pd.DataFrame(all_episode_rows, columns=columns_episodes).to_csv(os.path.join(OUTPUT_DIR,'episodes.csv'), index=False)
pd.DataFrame(all_step_records, columns=columns_steps).to_csv(os.path.join(OUTPUT_DIR,'steps.csv'), index=False)

print(f"Simulation Complete. CSVs saved.")

conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()

# Sync to DB
for solver_experiment in config['experiments']:
    exp_name = solver_experiment['name']
    exp_n = solver_experiment['N_episodes']
    
    cur.execute("INSERT OR REPLACE INTO Experiments (name, monitor_level, description, N_episodes) VALUES (?, ?, ?, ?)", (exp_name, 'SOLVER_UPGRADED', '', exp_n))
    experiment_id = cur.execute("SELECT experiment_id FROM Experiments WHERE name=?", (exp_name,)).fetchone()[0]
    
    exp_rows = [r for r in all_episode_rows if r[0] == exp_name]
    episode_records = [(experiment_id,) + row[2:] for row in exp_rows]
    
    cur.execute("DELETE FROM Episodes WHERE experiment_id=?", (experiment_id,))
    cur.executemany("INSERT INTO Episodes (experiment_id, run_idx, success, total_downtime, total_damage, total_defense_cost, total_falsepos_actions, mean_time_to_detect, total_attacker_utility) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", episode_records)
    conn.commit()
    
    df_ep_ids = pd.read_sql("SELECT episode_id, run_idx FROM Episodes WHERE experiment_id=?", conn, params=(experiment_id,))
    df_steps_all = pd.DataFrame(all_step_records, columns=columns_steps)
    df_steps_exp = df_steps_all[df_steps_all['experiment'] == exp_name].copy()
    df_merge = pd.merge(df_steps_exp, df_ep_ids, on='run_idx', how='inner')
    
    cur.execute("DELETE FROM Steps WHERE episode_id IN (SELECT episode_id FROM Episodes WHERE experiment_id=?)", (experiment_id,))
    
    step_records_final = []
    for _, row in df_merge.iterrows():
        step_records_final.append((
            row['episode_id'], row['step_idx'], row['ck_stage'], row['technique_id'], row['technique_name'],
            row['action_success'], row['detected'], row['detect_prob'], row['detection_delay'], row['impact_minutes'],
            row['attacker_belief_high_cap'], row['nash_prob_chosen'], row['discounted_payoff_step']
        ))
    cur.executemany("INSERT INTO Steps (episode_id, step_idx, ck_stage, technique_id, technique_name, action_success, detected, detect_prob, detection_delay, impact_minutes, attacker_belief_high_cap, nash_prob_chosen, discounted_payoff_step) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", step_records_final)
    conn.commit()

conn.close()
print(f"DB Insertion Complete for {len(config['experiments'])} experiments.")