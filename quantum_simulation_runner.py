#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:35:30 2025
    Create necessary directories for data and results
    
    Returns:
    --------
    list
        List of created directories
@author: mjp38
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from synthetic_data_generator import SyntheticWheatPolicyData
from quantum_hamiltonian_simplified import QuantumPolicyHamiltonian

def setup_directories():

    dirs = ['synthetic_data', 'quantum_results']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    return dirs

def generate_synthetic_data():
    """
    Generate synthetic wheat policy data
    
    Returns:
    --------
    SyntheticWheatPolicyData
        Data generator with synthetic data
    """
    print("Generating synthetic wheat policy data...")
    
    # Create data generator
    generator = SyntheticWheatPolicyData(start_year=2018, end_year=2024)
    
    # Generate all data
    generator.generate_all_data()
    
    # Export to CSV
    generator.export_to_csv("synthetic_data")
    
    return generator

def prepare_quantum_model(generator):
    """
    Prepare quantum policy Hamiltonian model
    
    Parameters:
    -----------
    generator : SyntheticWheatPolicyData
        Data generator with synthetic data
        
    Returns:
    --------
    QuantumPolicyHamiltonian
        Quantum policy model
    """
    print("Preparing quantum policy Hamiltonian model...")
    
    # Get latest year policy data
    latest_year = max(generator.years)
    policy_data = pd.DataFrame(index=generator.countries)
    
    # Extract relevant policy columns
    for country in generator.countries:
        if (country, latest_year) in generator.policy_data.index:
            policy = generator.policy_data.loc[(country, latest_year)]
            policy_data.loc[country, 'import_tariff_rate'] = policy['import_tariff_rate']
            policy_data.loc[country, 'direct_producer_subsidy'] = policy['direct_producer_subsidy']
            policy_data.loc[country, 'stock_to_use_ratio'] = policy['stock_to_use_ratio']
    
    # Calculate coupling matrices
    coupling_matrices = generator.calculate_policy_coupling_matrices()
    
    # Create quantum model
    model = QuantumPolicyHamiltonian(policy_data, coupling_matrices)
    
    # Export coupling matrices
    for name, matrix in coupling_matrices.items():
        matrix.to_csv(f"synthetic_data/{name}_coupling.csv")
    
    return model

def run_simulation(model, output_dir="quantum_results"):
    """
    Run quantum policy simulation
    
    Parameters:
    -----------
    model : QuantumPolicyHamiltonian
        Quantum policy model
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    tuple
        Tuple containing (policy_evolution, equilibrium)
    """
    print("Running quantum policy simulation...")
    
    # Run simulation
    policy_evolution = model.run_simulation()
    equilibrium = model.calculate_equilibrium()
    
    # Visualize results
    model.visualize_policy_evolution(policy_evolution, output_dir)
    model.visualize_coupling_matrices(output_dir)
    model.visualize_equilibrium(equilibrium, output_dir)
    
    # Export results
    policy_evolution.to_csv(f"{output_dir}/policy_evolution.csv")
    equilibrium.to_csv(f"{output_dir}/equilibrium.csv")
    
    return policy_evolution, equilibrium

def run_scenario_analysis(model, output_dir="quantum_results"):
    """
    Run scenario analysis
    
    Parameters:
    -----------
    model : QuantumPolicyHamiltonian
        Quantum policy model
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary containing results for each scenario
    """
    print("Running scenario analysis...")
    
    # Define scenarios
    scenarios = {
        'high_tariffs': {
            'USA': {'tariffs': 4},  # Increase US tariffs to maximum
            'EU': {'tariffs': 4},   # Increase EU tariffs to maximum
            'China': {'tariffs': 4}  # Increase China tariffs to maximum
        },
        'reduced_subsidies': {
            'USA': {'subsidies': 1},  # Reduce US subsidies
            'EU': {'subsidies': 2},   # Reduce EU subsidies
            'China': {'subsidies': 2}  # Reduce China subsidies
        },
        'increased_stockpiling': {
            'USA': {'stockpiling': 3},    # Increase US stockpiles
            'Russia': {'stockpiling': 3},  # Increase Russia stockpiles
            'India': {'stockpiling': 3}    # Increase India stockpiles
        },
        'trade_war': {
            'USA': {'tariffs': 4},       # Maximum US tariffs
            'China': {'tariffs': 4},     # Maximum China tariffs
            'EU': {'tariffs': 3},        # High EU tariffs
            'Russia': {'stockpiling': 3}, # Increased Russia stockpiling
            'India': {'subsidies': 4}     # Maximum India subsidies
        },
        'policy_coordination': {
            'USA': {'tariffs': 1, 'subsidies': 1},     # Low US tariffs and subsidies
            'EU': {'tariffs': 1, 'subsidies': 1},      # Low EU tariffs and subsidies
            'Russia': {'tariffs': 1, 'stockpiling': 1}, # Low Russia tariffs and stockpiling
            'China': {'tariffs': 1, 'stockpiling': 2},  # Low China tariffs, moderate stockpiling
            'India': {'tariffs': 1, 'subsidies': 1}     # Low India tariffs and subsidies
        }
    }
    
    # Run scenario comparison
    scenario_results = model.compare_scenarios(scenarios, output_dir)
    
    return scenario_results

def analyze_results(scenario_results, output_dir="quantum_results"):
    """
    Analyze simulation results
    
    Parameters:
    -----------
    scenario_results : dict
        Dictionary containing results for each scenario
    output_dir : str
        Directory to save results
    """
    print("Analyzing simulation results...")
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Extract equilibrium results for all scenarios
    equilibrium_comparison = pd.DataFrame()
    
    for scenario, results in scenario_results.items():
        equilibrium = results['equilibrium']
        
        # Add scenario column
        equilibrium_reset = equilibrium.reset_index()
        equilibrium_reset['scenario'] = scenario
        
        # Append to comparison dataframe
        equilibrium_comparison = pd.concat([equilibrium_comparison, equilibrium_reset])
    
    # Export comparison data
    equilibrium_comparison.to_csv(f"{analysis_dir}/equilibrium_comparison.csv")
    
    # Calculate policy stability metrics
    stability_metrics = pd.DataFrame(index=scenario_results.keys())
    
    for scenario, results in scenario_results.items():
        evolution = results['evolution']
        
        # Calculate policy volatility (standard deviation over time)
        volatility = evolution.groupby(['country', 'dimension'])['expected_level'].std()
        
        # Calculate average volatility for each dimension
        for dimension in ['tariffs', 'subsidies', 'stockpiling']:
            dim_volatility = volatility.xs(dimension, level='dimension').mean()
            stability_metrics.loc[scenario, f"{dimension}_volatility"] = dim_volatility
        
        # Calculate overall volatility
        stability_metrics.loc[scenario, 'overall_volatility'] = volatility.mean()
    
    # Export stability metrics
    stability_metrics.to_csv(f"{analysis_dir}/stability_metrics.csv")
    
    # Visualize stability metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    stability_metrics[[
        'tariffs_volatility', 'subsidies_volatility', 'stockpiling_volatility'
    ]].plot(kind='bar', ax=ax)
    
    ax.set_ylabel("Policy Volatility (Standard Deviation)")
    ax.set_title("Policy Stability Comparison Across Scenarios")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title="Policy Dimension")
    
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/policy_stability.png", dpi=300)
    
    print(f"Analysis complete. Results saved to {analysis_dir}/")

def main():
    """
    Main function to run the complete simulation pipeline
    """
    print("Starting Quantum Policy Hamiltonian Simulation")
    print("==============================================")
    
    # Setup directories
    dirs = setup_directories()
    print(f"Created directories: {', '.join(dirs)}")
    
    # Generate synthetic data
    generator = generate_synthetic_data()
    
    # Prepare quantum model
    model = prepare_quantum_model(generator)
    
    # Run baseline simulation
    policy_evolution, equilibrium = run_simulation(model)
    
    # Run scenario analysis
    scenario_results = run_scenario_analysis(model)
    
    # Analyze results
    analyze_results(scenario_results)
    
    print("\nQuantum Policy Hamiltonian Simulation Complete")
    print("==============================================")
    print("Results available in the following directories:")
    print("  - synthetic_data/: Synthetic wheat policy data")
    print("  - quantum_results/: Simulation results and visualizations")


if __name__ == "__main__":
    main()