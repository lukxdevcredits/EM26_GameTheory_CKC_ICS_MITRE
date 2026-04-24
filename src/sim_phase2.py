#!/usr/bin/env python3
"""
Phase-2 ICS Game-Theoretic SOLVER (BASELINE RATIONAL)
Author: LBD
Date: 2025
Description: Runs the "Baseline" simulation. 
             This model uses a simpler Rational Attacker (3x2 Game) 
             to provide a comparison point for the Advanced Solver (Phase 4).
"""

import json, os
import numpy as np
import nashpy as nash 
from math import exp, log

# -----------------------
# CONFIG & PATHS
# -----------------------
INPUT_DIR = '.'
OUTPUT_DIR = './Phase2_Outputs' 
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_JSON = os.path.join(INPUT_DIR, 'attack_techniques.json')
PAYOFF_JSON = os.path.join(INPUT_DIR, 'payoff_model.json')
CONFIG_JSON = os.path.join(INPUT_DIR, 'config_phase2.json')

# Load Config
with open(ATTACK_JSON,'r') as f:
    attack_techniques = json.load(f)
with open(PAYOFF_JSON,'r') as f:
    payoff_model = json.load(f)
with open(CONFIG_JSON,'r') as f:
    config = json.load(f)

# Unpack Params
P_CONF = payoff_model['system_parameters']
B_CONF = payoff_model['bayesian_parameters']
PO_CONF = payoff_model['payoff_parameters']
D_CONF = payoff_model['detection_parameters']

np.random.seed(config.get('seed', 12345))

CKC_STAGES = ["Reconnaissance", "Weaponization", "Delivery", "Exploitation", 
              "Installation", "Command & Control", "Actions on Objectives"]

class BayesianAttacker:
    """
    Represents an attacker who learns about the defender's capabilities over time.
    """
    def __init__(self):
        """
        Init the attacker's belief state.
        
        The attacker starts with a prob estimate (belief) regarding whether 
        the defender is 'High Capability' or 'Low Capability'.
        """
        self.belief = np.array([
            1.0 - B_CONF['initial_belief_high_cap'], 
            B_CONF['initial_belief_high_cap']
        ])
    
    def update_belief(self, defender_action):
        """
        Updates belief based on defender response.
        
        Params:
            defender_action (str): "Block" or "Passive"
            
        Logic:
            - Blocked? -> Increase belief in High Capability.
            - Passive? -> Lean towards Low Capability.
        """
        if defender_action == "Block":
            likelihood = np.array([
                B_CONF['likelihood_block_low_cap'], 
                B_CONF['likelihood_block_high_cap']
            ])
        else:
            likelihood = np.array([
                1.0 - B_CONF['likelihood_block_low_cap'], 
                1.0 - B_CONF['likelihood_block_high_cap']
            ])
            
        posterior = likelihood * self.belief
        P_E = np.sum(posterior)
        if P_E == 0: return 
        self.belief = posterior / P_E
        
    def get_belief_adjustment(self):
        """
        Calc penalty/bonus based on caution.
        
        Returns:
            float: Penalty (negative) if they think defender is strong.
                   Reduces incentive for aggression.
        """
        cautiousness_factor = (self.belief[1] - 0.5) * 2 
        if self.belief[1] > 0.5:
            return PO_CONF['attacker_aggressive_gain'] * -0.5 * cautiousness_factor
        return 0.0

def create_payoff_matrix(stage, belief_adjustment):
    """
    Constructs the 3x2 Game Matrix.
    
    Params:
        stage (str): Current CKC stage
        belief_adjustment (float): Modifier from belief state
        
    Returns:
        nash.Game: Object with matrices for Attacker & Defender
        
    Logic:
        - Rows: Attacker (Stealthy, Std, Aggro)
        - Cols: Defender (Passive, Block)
        - Values: Base gains * Stage Multipliers
    """
    A = np.zeros((3, 2))
    # Attacker Payoffs
    A[0, 0] = PO_CONF['attacker_stealthy_gain'] * stage_multiplier(stage)  
    A[0, 1] = PO_CONF['penalty_stealthy_detection']                      
    A[1, 0] = PO_CONF['attacker_standard_gain'] * stage_multiplier(stage) 
    A[1, 1] = PO_CONF['penalty_stealthy_detection'] * 2                  
    A[2, 0] = (PO_CONF['attacker_aggressive_gain'] + belief_adjustment) * stage_multiplier(stage) 
    A[2, 1] = PO_CONF['penalty_aggressive_detection']                    
    
    # Defender Payoffs (Zero-sum ish)
    D = np.zeros((3, 2))
    D[0, 0] = -A[0, 0] + PO_CONF['base_defender_cost_passive']
    D[1, 0] = -A[1, 0] + PO_CONF['base_defender_cost_passive']
    D[2, 0] = -A[2, 0] + PO_CONF['base_defender_cost_passive']
    D[0, 1] = -A[0, 1] + PO_CONF['base_defender_cost_active'] * 0.5
    D[1, 1] = -A[1, 1] + PO_CONF['base_defender_cost_active'] 
    D[2, 1] = -A[2, 1] + PO_CONF['base_defender_cost_active'] * 2.0
    
    # Add tiny noise to avoid solver convergence errors
    epsilon = 1e-9
    A += np.random.uniform(-epsilon, epsilon, A.shape)
    D += np.random.uniform(-epsilon, epsilon, D.shape)

    return nash.Game(A, -D) 

def solve_nash(game):
    """
    Finds Nash Equilibrium.
    
    Params:
        game (nash.Game): The game object
        
    Returns:
        (att_probs, def_probs): Optimal mixed strategies
    """
    equilibria = game.support_enumeration()
    try:
        return next(equilibria)
    except StopIteration:
        # Fallback to defaults if solver fails
        return (np.array([1/3, 1/3, 1/3]), np.array([0.5, 0.5])) 

def stage_multiplier(stage):
    """
    Returns weight factor for the stage.
    
    Logic:
        Later stages (Actions on Objectives) are worth more (1.0) 
        than early ones (Recon = 0.1).
    """
    multipliers = {
        "Reconnaissance": 0.1, "Weaponization": 0.2, "Delivery": 0.4, 
        "Exploitation": 0.6, "Installation": 0.7, "Command & Control": 0.8,
        "Actions on Objectives": 1.0
    }
    return multipliers.get(stage, 0.5)

def run_solver_simulation():
    """
    Executes Main Simulation Loop.
    
    Process:
        1. Loop Episodes
        2. Loop CKC Stages
        3. Solve Game Theory Matrix
        4. Record Outcome -> CSV
    """
    all_episode_rows = []
    all_step_records = []
    
    EXP_NAME = "Baseline_Rational"
    N_EPISODES = 200 
    
    print(f"Running Baseline Rational Simulation ({N_EPISODES} episodes)...")
    
    for run_idx in range(N_EPISODES):
        attacker = BayesianAttacker()
        episode_success = 0
        current_time_step = 0
        current_discount = 1.0
        total_utility = 0.0
        
        for step_idx, stage_name in enumerate(CKC_STAGES): 
            # 1. Update Game Params (based on belief)
            belief_adj = attacker.get_belief_adjustment()
            game = create_payoff_matrix(stage_name, belief_adj)
            
            # 2. Solve Nash
            att_probs, def_probs = solve_nash(game)
            
            # 3. Select Actions (Weighted Random)
            att_choice_idx = np.random.choice(len(att_probs), p=att_probs)
            att_strategy = ["Stealthy", "Standard", "Aggressive"][att_choice_idx]
            
            # 4. Filter Techniques
            available_techs = [t for group in attack_techniques['ics_attack_tactics'] 
                               if group['ckc_stage'] == stage_name
                               for t in group['techniques'] if t['strategy_class'] == att_strategy]
            
            if not available_techs: continue 
            chosen_tech = np.random.choice(available_techs)
            
            def_choice_idx = np.random.choice(len(def_probs), p=def_probs)
            def_action = ["Passive", "Block"][def_choice_idx]
            
            # 5. Determine Outcome
            detected = (def_action == "Block")
            
            if detected:
                attacker.update_belief("Block")
                episode_success = 0 
                break 
            
            action_success = np.random.rand() < chosen_tech['base_success_prob']
            
            if stage_name == "Actions on Objectives" and action_success:
                episode_success = 1
            
            if not detected and not action_success:
                 attacker.update_belief("Passive")

            # 6. Calc Payoffs
            att_payoff_step = game.payoff_matrices[0][att_choice_idx, def_choice_idx] * current_discount
            total_utility += att_payoff_step

            # 7. Log Data
            all_step_records.append((
                EXP_NAME, run_idx, step_idx, stage_name, 
                chosen_tech['id'], chosen_tech['name'], action_success, 
                detected, chosen_tech['base_detectability_lambda'], current_time_step, 
                chosen_tech['impact_magnitude'],
                attacker.belief[1],  
                att_probs[att_choice_idx], 
                att_payoff_step 
            ))
            
            current_time_step += 1
            current_discount *= P_CONF['discount_factor']
        
        all_episode_rows.append((
            EXP_NAME, config['seed'], run_idx, episode_success, 
            0, 0, 0, 0, 0, 
            total_utility 
        ))
    
    # Save Results
    columns_episodes = ['experiment','seed','run_idx','success','total_downtime','total_damage',
                        'total_defense_cost','total_falsepos_actions','mean_time_to_detect',
                        'total_attacker_utility'] 
    
    columns_steps = ['experiment','run_idx','step_idx','ck_stage','technique_id','technique_name',
                     'action_success','detected','detect_prob','detection_delay','impact_minutes',
                     'attacker_belief_high_cap', 'nash_prob_chosen', 'discounted_payoff_step'] 
    
    import pandas as pd
    pd.DataFrame(all_episode_rows, columns=columns_episodes).to_csv(os.path.join(OUTPUT_DIR,'episodes.csv'), index=False)
    pd.DataFrame(all_step_records, columns=columns_steps).to_csv(os.path.join(OUTPUT_DIR,'steps.csv'), index=False)

    print(f"Baseline Simulation Complete. Results saved as '{EXP_NAME}'.")

if __name__ == "__main__":
    run_solver_simulation()