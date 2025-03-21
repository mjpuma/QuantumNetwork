#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:34:01 2025

    Simplified implementation of a quantum policy Hamiltonian for wheat market policy analysis
    for the 5-country model (USA, EU, Russia, China, India)

@author: mjp38
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qutip import sigmaz, sigmax, identity, tensor, basis, Qobj
import os

class QuantumPolicyHamiltonian:

    
    def __init__(self, policy_data=None, coupling_matrices=None):
        """
        Initialize the quantum Hamiltonian model
        
        Parameters:
        -----------
        policy_data : pandas.DataFrame
            Policy data for the 5 countries
        coupling_matrices : dict
            Dictionary of coupling matrices for different policy dimensions
        """
        # Countries in the model
        self.countries = ['USA', 'EU', 'Russia', 'China', 'India']
        
        # Policy dimensions to model
        self.dimensions = ['tariffs', 'subsidies', 'stockpiling']
        
        # Store policy data and coupling matrices
        self.policy_data = policy_data
        self.coupling_matrices = coupling_matrices
        
        # Parameters for the Hamiltonian
        self.h_local = None
        self.h_coupling = None
        self.h_total = None
        
        # Parameters for time evolution
        self.time_steps = 20
        self.dt = 0.1
        self.hbar = 1.0
        
        # Policy level discretization (5 levels: 0, 1, 2, 3, 4)
        self.levels = 5
    
    def load_data(self, policy_file, tariff_file, subsidy_file, stockpile_file):
        """
        Load policy data and coupling matrices from CSV files
        
        Parameters:
        -----------
        policy_file : str
            Path to policy data CSV file
        tariff_file : str
            Path to tariff coupling matrix CSV file
        subsidy_file : str
            Path to subsidy coupling matrix CSV file
        stockpile_file : str
            Path to stockpile coupling matrix CSV file
        """
        try:
            # Load policy data
            self.policy_data = pd.read_csv(policy_file, index_col=0)
            
            # Load coupling matrices
            tariff_coupling = pd.read_csv(tariff_file, index_col=0)
            subsidy_coupling = pd.read_csv(subsidy_file, index_col=0)
            stockpile_coupling = pd.read_csv(stockpile_file, index_col=0)
            
            self.coupling_matrices = {
                'tariffs': tariff_coupling,
                'subsidies': subsidy_coupling,
                'stockpiling': stockpile_coupling
            }
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def normalize_policy_data(self):
        """
        Normalize policy data to discrete levels (0-4)
        
        Returns:
        --------
        pandas.DataFrame
            Normalized policy data with discrete levels
        """
        # Mapping between policy dimensions and data columns
        column_mapping = {
            'tariffs': 'import_tariff_rate',
            'subsidies': 'direct_producer_subsidy',
            'stockpiling': 'stock_to_use_ratio'
        }
        
        # Normalization ranges for each dimension
        norm_ranges = {
            'import_tariff_rate': (0, 100),      # %
            'direct_producer_subsidy': (0, 100), # USD/ton
            'stock_to_use_ratio': (0, 50)        # %
        }
        
        # Create normalized policy dataframe
        policy_norm = pd.DataFrame(index=self.countries)
        
        # Perform normalization
        for dimension, column in column_mapping.items():
            if column in self.policy_data.columns:
                min_val, max_val = norm_ranges[column]
                
                # Normalize values to 0-1 range
                normalized = (self.policy_data[column] - min_val) / (max_val - min_val)
                normalized = normalized.clip(0, 1)  # Ensure 0-1 range
                
                # Convert to discrete levels (0-4)
                policy_norm[dimension] = (normalized * (self.levels - 1)).round().astype(int)
            else:
                # Default values if column not present
                policy_norm[dimension] = 2  # Middle level (2 out of 0-4)
        
        return policy_norm
    
    def build_hamiltonian(self):
        """
        Build the quantum Hamiltonian for the policy model
        
        Returns:
        --------
        qutip.Qobj
            Quantum Hamiltonian for the system
        """
        # Check if data is loaded
        if self.policy_data is None or self.coupling_matrices is None:
            raise ValueError("Policy data and coupling matrices must be loaded first")
        
        # Normalize policy data to discrete levels
        policy_norm = self.normalize_policy_data()
        
        # System dimensions
        n_countries = len(self.countries)
        n_dimensions = len(self.dimensions)
        
        # Pauli matrices
        sigma_z = sigmaz()
        sigma_x = sigmax()
        
        # Transverse field strength
        gamma = 0.5
        
        # Create local field terms (current policy state)
        h_local_list = []
        for dim_idx, dimension in enumerate(self.dimensions):
            for country_idx, country in enumerate(self.countries):
                # Current policy level
                level = policy_norm.loc[country, dimension]
                
                # Convert to field strength (-1 to 1)
                field = 2 * (level / (self.levels - 1)) - 1
                
                # Create operator list for tensor product
                op_list = []
                for i in range(n_countries * n_dimensions):
                    if i == dim_idx + country_idx * n_dimensions:
                        op_list.append(sigma_z)
                    else:
                        op_list.append(identity(2))
                
                # Create tensor product operator
                h_term = field * tensor(op_list)
                h_local_list.append(h_term)
        
        # Sum all local field terms
        h_local = sum(h_local_list)
        
        # Create coupling terms (interactions between countries)
        h_coupling_list = []
        for dim_idx, dimension in enumerate(self.dimensions):
            # Get coupling matrix for this dimension
            coupling_matrix = self.coupling_matrices[dimension].values
            
            # Add coupling terms between countries
            for i in range(n_countries):
                for j in range(i+1, n_countries):  # only upper triangle to avoid redundancy
                    # Get coupling strength
                    J_ij = coupling_matrix[i, j]
                    
                    # Skip if coupling is zero
                    if J_ij == 0:
                        continue
                    
                    # Create operator lists for tensor product
                    op_list_i = []
                    op_list_j = []
                    for k in range(n_countries * n_dimensions):
                        country_k = k // n_dimensions
                        dim_k = k % n_dimensions
                        
                        # For first operator
                        if country_k == i and dim_k == dim_idx:
                            op_list_i.append(sigma_z)
                        else:
                            op_list_i.append(identity(2))
                        
                        # For second operator
                        if country_k == j and dim_k == dim_idx:
                            op_list_j.append(sigma_z)
                        else:
                            op_list_j.append(identity(2))
                    
                    # Create tensor product operators
                    h_term_i = tensor(op_list_i)
                    h_term_j = tensor(op_list_j)
                    
                    # Add interaction term
                    h_coupling_list.append(J_ij * h_term_i * h_term_j)
        
        # Create transverse field terms for quantum fluctuations
        h_trans_list = []
        for i in range(n_countries * n_dimensions):
            # Create operator list for tensor product
            op_list = []
            for j in range(n_countries * n_dimensions):
                if j == i:
                    op_list.append(sigma_x)
                else:
                    op_list.append(identity(2))
            
            # Create tensor product operator
            h_term = gamma * tensor(op_list)
            h_trans_list.append(h_term)
        
        # Sum all coupling terms
        h_coupling = sum(h_coupling_list)
        
        # Sum all transverse field terms
        h_trans = sum(h_trans_list)
        
        # Total Hamiltonian
        h_total = h_local + h_coupling + h_trans
        
        # Store Hamiltonian components
        self.h_local = h_local
        self.h_coupling = h_coupling
        self.h_trans = h_trans
        self.h_total = h_total
        
        return h_total
    
    def create_initial_state(self):
        """
        Create initial quantum state based on current policy levels
        
        Returns:
        --------
        qutip.Qobj
            Initial quantum state
        """
        # Normalize policy data to discrete levels
        policy_norm = self.normalize_policy_data()
        
        # System dimensions
        n_countries = len(self.countries)
        n_dimensions = len(self.dimensions)
        
        # Create individual state vectors
        state_list = []
        for country in self.countries:
            for dimension in self.dimensions:
                # Current policy level
                level = policy_norm.loc[country, dimension]
                
                # Convert to quantum state
                # Level 0 -> |0⟩, Level 4 -> |1⟩, other levels are superpositions
                alpha = np.sqrt(1 - level / (self.levels - 1))
                beta = np.sqrt(level / (self.levels - 1))
                
                # Add state vector to list
                state = alpha * basis(2, 0) + beta * basis(2, 1)
                state_list.append(state)
        
        # Create tensor product of all states
        initial_state = tensor(state_list)
        
        return initial_state
    
    def time_evolve(self):
        """
        Perform time evolution of the policy state
        
        Returns:
        --------
        list
            List of state vectors at each time step
        """
        # Build Hamiltonian if needed
        if self.h_total is None:
            self.build_hamiltonian()
        
        # Create initial state
        initial_state = self.create_initial_state()
        
        # Time evolution
        states = [initial_state]
        
        # Time evolution operator
        U = (-1j * self.h_total * self.dt / self.hbar).expm()
        
        # Evolve state
        current_state = initial_state
        for t in range(self.time_steps):
            current_state = U * current_state
            states.append(current_state)
        
        return states
    
    def extract_policy_expectations(self, evolved_states):
        """
        Extract expected policy levels from evolved quantum states
        
        Parameters:
        -----------
        evolved_states : list
            List of state vectors at each time step
            
        Returns:
        --------
        pandas.DataFrame
            Expected policy levels for each country, dimension, and time step
        """
        # System dimensions
        n_countries = len(self.countries)
        n_dimensions = len(self.dimensions)
        
        # Create multi-index for results
        index = pd.MultiIndex.from_product(
            [range(len(evolved_states)), self.countries, self.dimensions],
            names=['time_step', 'country', 'dimension']
        )
        
        # Initialize results dataframe
        results = pd.DataFrame(index=index, columns=['expected_level'])
        
        # Calculate expectation values
        for t, state in enumerate(evolved_states):
            for country_idx, country in enumerate(self.countries):
                for dim_idx, dimension in enumerate(self.dimensions):
                    # Index in the combined system
                    i = country_idx * n_dimensions + dim_idx
                    
                    # Create measurement operator (sigma_z)
                    op_list = []
                    for j in range(n_countries * n_dimensions):
                        if j == i:
                            op_list.append(sigma_z)
                        else:
                            op_list.append(identity(2))
                    
                    # Create tensor product operator
                    measure_op = tensor(op_list)
                    
                    # Calculate expectation value
                    expectation = (state.dag() * measure_op * state).tr().real
                    
                    # Convert from [-1,1] range to [0,4] discrete levels
                    expected_level = ((expectation + 1) / 2) * (self.levels - 1)
                    
                    # Store result
                    results.loc[(t, country, dimension), 'expected_level'] = expected_level
        
        return results
    
    def run_simulation(self):
        """
        Run a complete policy simulation
        
        Returns:
        --------
        pandas.DataFrame
            Policy evolution data
        """
        # Build Hamiltonian
        self.build_hamiltonian()
        
        # Run time evolution
        evolved_states = self.time_evolve()
        
        # Extract policy expectations
        policy_evolution = self.extract_policy_expectations(evolved_states)
        
        return policy_evolution
    
    def calculate_equilibrium(self):
        """
        Calculate policy equilibrium by finding the ground state
        
        Returns:
        --------
        pandas.DataFrame
            Equilibrium policy levels
        """
        # Build Hamiltonian if needed
        if self.h_total is None:
            self.build_hamiltonian()
        
        # Find ground state of Hamiltonian
        eigvals, eigvecs = self.h_total.eigenstates(sparse=True, sort='low')
        ground_state = eigvecs[0]
        
        # Extract policy expectations from ground state
        results = self.extract_policy_expectations([ground_state])
        
        # Filter for time_step=0 (the only one in this case)
        equilibrium = results.loc[0]
        
        # Reshape to have countries as rows and dimensions as columns
        equilibrium_df = equilibrium.reset_index().pivot(
            index='country', 
            columns='dimension', 
            values='expected_level'
        )
        
        return equilibrium_df
    
    def visualize_policy_evolution(self, results, save_dir=None):
        """
        Visualize the evolution of policy levels
        
        Parameters:
        -----------
        results : pandas.DataFrame
            Policy evolution data from run_simulation
        save_dir : str, optional
            Directory to save visualizations
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with policy evolution plots
        """
        # Reshape data for plotting
        plot_data = results.reset_index()
        
        # Create figure with subplots for each dimension
        fig, axes = plt.subplots(
            len(self.dimensions), 1, 
            figsize=(10, 4 * len(self.dimensions)),
            sharex=True
        )
        
        # If only one dimension, axes will not be an array
        if len(self.dimensions) == 1:
            axes = [axes]
        
        # Plot each dimension
        for dim_idx, dimension in enumerate(self.dimensions):
            ax = axes[dim_idx]
            
            # Filter data for this dimension
            dim_data = plot_data[plot_data['dimension'] == dimension]
            
            # Plot each country
            for country in self.countries:
                country_data = dim_data[dim_data['country'] == country]
                ax.plot(
                    country_data['time_step'],
                    country_data['expected_level'],
                    label=country,
                    marker='o',
                    linewidth=2
                )
            
            # Set labels and title
            ax.set_ylabel(f"{dimension.capitalize()} Policy Level (0-4)")
            ax.set_title(f"{dimension.capitalize()} Policy Evolution")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim([-0.1, 4.1])  # Add a little padding
        
        # Set x-axis label for the bottom subplot
        axes[-1].set_xlabel("Time Step")
        
        # Add overall title
        plt.suptitle("Quantum Policy Evolution Over Time", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if requested
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/policy_evolution.png", dpi=300)
        
        return fig
    
    def visualize_coupling_matrices(self, save_dir=None):
        """
        Visualize the coupling matrices
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save visualizations
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with coupling matrix heatmaps
        """
        # Check if coupling matrices are loaded
        if self.coupling_matrices is None:
            raise ValueError("Coupling matrices must be loaded first")
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            1, len(self.dimensions), 
            figsize=(5 * len(self.dimensions), 4),
            squeeze=False
        )
        
        # Plot each coupling matrix
        for dim_idx, dimension in enumerate(self.dimensions):
            ax = axes[0, dim_idx]
            
            # Get coupling matrix
            matrix = self.coupling_matrices[dimension]
            
            # Create heatmap
            sns.heatmap(
                matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                ax=ax,
                cbar_kws={'label': 'Coupling Strength'}
            )
            
            # Set title
            ax.set_title(f"{dimension.capitalize()} Coupling Matrix")
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/coupling_matrices.png", dpi=300)
        
        return fig
    
    def visualize_equilibrium(self, equilibrium_df, save_dir=None):
        """
        Visualize the equilibrium policy levels
        
        Parameters:
        -----------
        equilibrium_df : pandas.DataFrame
            Equilibrium policy levels from calculate_equilibrium
        save_dir : str, optional
            Directory to save visualizations
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with equilibrium policy levels
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot equilibrium policy levels
        equilibrium_df.plot(kind='bar', ax=ax)
        
        # Set labels and title
        ax.set_ylabel("Equilibrium Policy Level (0-4)")
        ax.set_title("Equilibrium Policy Levels by Country")
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title="Policy Dimension")
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/equilibrium_policies.png", dpi=300)
        
        return fig
    
    def run_scenario(self, scenario_name, policy_changes, save_dir=None):
        """
        Run a policy scenario with modified initial conditions
        
        Parameters:
        -----------
        scenario_name : str
            Name of the scenario
        policy_changes : dict
            Dictionary of policy changes, e.g., {'USA': {'tariffs': 4}}
        save_dir : str, optional
            Directory to save results
            
        Returns:
        --------
        tuple
            Tuple containing (policy_evolution, equilibrium)
        """
        # Make a copy of the original policy data
        original_policy = self.policy_data.copy()
        
        # Apply policy changes
        for country, changes in policy_changes.items():
            for dimension, level in changes.items():
                # Map dimension to column name
                column_mapping = {
                    'tariffs': 'import_tariff_rate',
                    'subsidies': 'direct_producer_subsidy',
                    'stockpiling': 'stock_to_use_ratio'
                }
                
                if dimension in column_mapping:
                    column = column_mapping[dimension]
                    
                    # Convert level (0-4) to actual value
                    if column == 'import_tariff_rate':
                        # Level 0: 0%, Level 4: 100%
                        value = level * 25
                    elif column == 'direct_producer_subsidy':
                        # Level 0: 0 USD/ton, Level 4: 100 USD/ton
                        value = level * 25
                    elif column == 'stock_to_use_ratio':
                        # Level 0: 0%, Level 4: 50%
                        value = level * 12.5
                    
                    # Update policy data
                    self.policy_data.loc[country, column] = value
        
        # Run simulation with modified policies
        self.h_total = None  # Reset Hamiltonian to rebuild with new policies
        policy_evolution = self.run_simulation()
        equilibrium = self.calculate_equilibrium()
        
        # Save results if requested
        if save_dir is not None:
            scenario_dir = os.path.join(save_dir, f"scenario_{scenario_name}")
            if not os.path.exists(scenario_dir):
                os.makedirs(scenario_dir)
            
            # Save policy evolution
            policy_evolution.to_csv(f"{scenario_dir}/policy_evolution.csv")
            
            # Save equilibrium
            equilibrium.to_csv(f"{scenario_dir}/equilibrium.csv")
            
            # Visualize policy evolution
            self.visualize_policy_evolution(policy_evolution, scenario_dir)
            
            # Visualize equilibrium
            self.visualize_equilibrium(equilibrium, scenario_dir)
        
        # Restore original policy data
        self.policy_data = original_policy
        self.h_total = None  # Reset Hamiltonian
        
        return policy_evolution, equilibrium
    
    def compare_scenarios(self, scenarios, save_dir=None):
        """
        Run and compare multiple policy scenarios
        
        Parameters:
        -----------
        scenarios : dict
            Dictionary of scenario definitions, e.g., {'scenario1': {'USA': {'tariffs': 4}}}
        save_dir : str, optional
            Directory to save results
            
        Returns:
        --------
        dict
            Dictionary containing results for each scenario
        """
        # Run baseline scenario first
        print("Running baseline scenario...")
        baseline_evolution = self.run_simulation()
        baseline_equilibrium = self.calculate_equilibrium()
        
        # Store results
        results = {
            'baseline': {
                'evolution': baseline_evolution,
                'equilibrium': baseline_equilibrium
            }
        }
        
        # Run each scenario
        for scenario_name, policy_changes in scenarios.items():
            print(f"Running scenario: {scenario_name}")
            evolution, equilibrium = self.run_scenario(scenario_name, policy_changes)
            
            # Store results
            results[scenario_name] = {
                'evolution': evolution,
                'equilibrium': equilibrium
            }
        
        # Compare results
        if save_dir is not None:
            compare_dir = os.path.join(save_dir, "scenario_comparison")
            if not os.path.exists(compare_dir):
                os.makedirs(compare_dir)
            
            # Compare equilibrium policies
            self.visualize_scenario_equilibrium_comparison(results, compare_dir)
            
            # Compare policy evolution for each dimension
            self.visualize_scenario_evolution_comparison(results, compare_dir)
        
        return results
    
    def visualize_scenario_equilibrium_comparison(self, results, save_dir):
        """
        Visualize comparison of equilibrium policies across scenarios
        
        Parameters:
        -----------
        results : dict
            Dictionary containing results for each scenario
        save_dir : str
            Directory to save visualizations
        """
        # Create figure for each dimension
        for dimension in self.dimensions:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Extract data for each scenario
            data = {}
            for scenario, scenario_results in results.items():
                equilibrium = scenario_results['equilibrium']
                data[scenario] = equilibrium[dimension]
            
            # Convert to DataFrame for plotting
            df = pd.DataFrame(data)
            
            # Plot
            df.plot(kind='bar', ax=ax)
            
            # Set labels and title
            ax.set_ylabel("Equilibrium Policy Level (0-4)")
            ax.set_title(f"{dimension.capitalize()} Policy Equilibrium Comparison")
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(title="Scenario")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{dimension}_equilibrium_comparison.png", dpi=300)
            plt.close()
    
    def visualize_scenario_evolution_comparison(self, results, save_dir):
        """
        Visualize comparison of policy evolution across scenarios
        
        Parameters:
        -----------
        results : dict
            Dictionary containing results for each scenario
        save_dir : str
            Directory to save visualizations
        """
        # Create figure for each country and dimension
        for country in self.countries:
            for dimension in self.dimensions:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract data for each scenario
                for scenario, scenario_results in results.items():
                    evolution = scenario_results['evolution']
                    
                    # Filter data for this country and dimension
                    mask = (evolution.index.get_level_values('country') == country) & \
                           (evolution.index.get_level_values('dimension') == dimension)
                    
                    data = evolution[mask].reset_index()
                    
                    # Plot
                    ax.plot(
                        data['time_step'],
                        data['expected_level'],
                        label=scenario,
                        marker='o',
                        linewidth=2
                    )
                
                # Set labels and title
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Policy Level (0-4)")
                ax.set_title(f"{country}: {dimension.capitalize()} Policy Evolution")
                ax.grid(True, alpha=0.3)
                ax.legend(title="Scenario")
                ax.set_ylim([-0.1, 4.1])  # Add a little padding
                
                # Save figure
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{country}_{dimension}_evolution_comparison.png", dpi=300)
                plt.close()


def main():
    """
    Main function to demonstrate the quantum policy Hamiltonian
    """
    # Create output directory
    output_dir = "quantum_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load sample policy data
    sample_policy_data = pd.DataFrame(
        index=['USA', 'EU', 'Russia', 'China', 'India'],
        data={
            'import_tariff_rate': [3.0, 60.0, 5.0, 65.0, 80.0],
            'direct_producer_subsidy': [40.0, 80.0, 30.0, 75.0, 55.0],
            'stock_to_use_ratio': [35.0, 15.0, 14.0, 50.0, 15.0]
        }
    )
    
    # Create sample coupling matrices
    tariff_coupling = pd.DataFrame(
        index=['USA', 'EU', 'Russia', 'China', 'India'],
        columns=['USA', 'EU', 'Russia', 'China', 'India'],
        data=[
            [1.0, -0.3, -0.2, -0.4, -0.2],
            [-0.3, 1.0, -0.5, -0.3, -0.2],
            [-0.2, -0.5, 1.0, -0.3, -0.4],
            [-0.4, -0.3, -0.3, 1.0, -0.3],
            [-0.2, -0.2, -0.4, -0.3, 1.0]
        ]
    )
    
    subsidy_coupling = pd.DataFrame(
        index=['USA', 'EU', 'Russia', 'China', 'India'],
        columns=['USA', 'EU', 'Russia', 'China', 'India'],
        data=[
            [1.0, -0.4, -0.3, -0.2, -0.3],
            [-0.4, 1.0, -0.4, -0.3, -0.2],
            [-0.3, -0.4, 1.0, -0.3, -0.4],
            [-0.2, -0.3, -0.3, 1.0, -0.4],
            [-0.3, -0.2, -0.4, -0.4, 1.0]
        ]
    )
    
    stockpile_coupling = pd.DataFrame(
        index=['USA', 'EU', 'Russia', 'China', 'India'],
        columns=['USA', 'EU', 'Russia', 'China', 'India'],
        data=[
            [1.0, -0.2, -0.2, -0.5, -0.3],
            [-0.2, 1.0, -0.3, -0.4, -0.2],
            [-0.2, -0.3, 1.0, -0.4, -0.4],
            [-0.5, -0.4, -0.4, 1.0, -0.5],
            [-0.3, -0.2, -0.4, -0.5, 1.0]
        ]
    )
    
    coupling_matrices = {
        'tariffs': tariff_coupling,
        'subsidies': subsidy_coupling,
        'stockpiling': stockpile_coupling
    }
    
    # Create quantum model
    model = QuantumPolicyHamiltonian(sample_policy_data, coupling_matrices)
    
    # Run baseline simulation
    print("Running baseline simulation...")
    policy_evolution = model.run_simulation()
    equilibrium = model.calculate_equilibrium()
    
    # Visualize results
    model.visualize_policy_evolution(policy_evolution, output_dir)
    model.visualize_coupling_matrices(output_dir)
    model.visualize_equilibrium(equilibrium, output_dir)
    
    # Export results
    policy_evolution.to_csv(f"{output_dir}/policy_evolution.csv")
    equilibrium.to_csv(f"{output_dir}/equilibrium.csv")
    
    # Define scenarios
    scenarios = {
        'high_tariffs': {
            'USA': {'tariffs': 4},  # Increase US tariffs to maximum
            'EU': {'tariffs': 4},   # Increase EU tariffs to maximum
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
        }
    }
    
    # Run scenario comparison
    print("Running scenario analysis...")
    scenario_results = model.compare_scenarios(scenarios, output_dir)
    
    print(f"Quantum simulation complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()