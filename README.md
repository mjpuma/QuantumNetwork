# Quantum Policy Hamiltonian Model for Wheat Markets

This repository contains a streamlined implementation of a quantum policy Hamiltonian framework for analyzing wheat market policies across five major countries (USA, EU, Russia, China, India). The model uses synthetic data to parameterize a quantum Hamiltonian and simulate the complex interactions between different policy dimensions.

## Overview

The Quantum Policy Hamiltonian framework treats national wheat market policies as quantum states that evolve over time based on coupling between different countries and policy dimensions. This approach allows for modeling complex interdependencies, path dependencies, and quantum effects in the global policy landscape.

## Components

The framework consists of three main components:

1. **Synthetic Data Generator** - Creates realistic synthetic policy data
2. **Quantum Hamiltonian Model** - Implements the quantum policy model
3. **Simulation Runner** - Coordinates data generation and simulations

## Synthetic Data Specifications

The synthetic data generator creates the following datasets:

### Policy Data

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| `import_tariff_rate` | Import tariff rate on wheat | % | 0-100 |
| `export_tax_rate` | Export tax or restriction | % | 0-25 |
| `direct_producer_subsidy` | Direct payments to producers | USD/ton | 0-100 |
| `price_support_level` | Price support level | USD/ton | 150-400 |
| `public_stockpile_target` | Target level for public stocks | MMT | 5-140 |
| `stock_to_use_ratio` | Stock-to-use ratio | % | 5-50 |
| `policy_intensity_index` | Composite index of policy intervention | unitless | 0-100 |

### Production Data

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| `area_harvested` | Harvested area for wheat | million hectares | 15-35 |
| `yield` | Wheat yield | tons/hectare | 2.5-6.0 |
| `production_volume` | Wheat production | million metric tons (MMT) | 50-150 |
| `domestic_consumption` | Domestic consumption | MMT | 30-150 |
| `feed_use` | Wheat used for animal feed | MMT | 3-50 |
| `food_use` | Wheat used for human food | MMT | 20-120 |
| `exports` | Wheat exports | MMT | 0-50 |
| `imports` | Wheat imports | MMT | 0-20 |
| `beginning_stocks` | Opening stocks | MMT | 5-150 |
| `ending_stocks` | Closing stocks | MMT | 5-150 |

### Trade Data

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| `wheat_volume` | Volume of wheat traded | MMT | 0-10 |
| `wheat_value` | Value of wheat traded | USD millions | 0-3000 |
| `average_price` | Average price of wheat | USD/ton | 200-350 |

### Stockpile Data

| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| `total_wheat_stocks` | Total wheat stocks | MMT | 5-150 |
| `public_wheat_stocks` | Public wheat stocks | MMT | 3-140 |
| `private_wheat_stocks` | Private wheat stocks | MMT | 2-50 |
| `target_stock_level` | Target stock level | MMT | 5-140 |
| `annual_stockpile_change` | Annual change in stockpile | MMT | -20 to 20 |
| `stockpile_acquisition_price` | Price paid to acquire stocks | USD/ton | 230-320 |
| `stockpile_release_price` | Price charged when releasing stocks | USD/ton | 220-340 |
| `storage_cost` | Annual storage cost | USD millions | 50-5000 |
| `storage_loss_rate` | Annual loss rate | % | 0.5-5.0 |
| `stocks_to_domestic_use` | Stocks to domestic use ratio | % | 5-100 |
| `months_of_consumption` | Months of consumption in storage | months | 0.5-12 |
| `price_stabilization_effect` | Estimated price stabilization effect | % | 5-50 |

### Coupling Matrices

Three coupling matrices are generated to represent policy interactions:

1. **Tariff Coupling Matrix** - How tariff policies interact between countries
2. **Subsidy Coupling Matrix** - How subsidy policies interact between countries
3. **Stockpile Coupling Matrix** - How stockpiling policies interact between countries

Values in these matrices range from -0.5 to 1.0, where:
- 1.0 represents strong self-coupling (diagonal elements)
- Positive values represent policy alignment (similar policies reinforce each other)
- Negative values represent policy opposition (policies counteract each other)
- Stronger coupling (larger magnitude) indicates stronger influence

## Quantum Model Parameters

The quantum Hamiltonian model uses the following parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `levels` | Number of discrete policy levels | 5 (0-4) |
| `time_steps` | Number of time steps for evolution | 20 |
| `dt` | Time step size | 0.1 |
| `hbar` | Planck's constant (scaled) | 1.0 |
| `gamma` | Transverse field strength | 0.5 |

## Policy Scenarios

The framework includes several predefined policy scenarios for analysis:

1. **Baseline** - Current policies based on synthetic data
2. **High Tariffs** - Increased tariffs in major importing countries
3. **Reduced Subsidies** - Decreased producer subsidies in major countries
4. **Increased Stockpiling** - Expanded strategic reserves in key countries
5. **Trade War** - High tariffs, export restrictions, and defensive stockpiling
6. **Policy Coordination** - Harmonized policies with reduced barriers

## Installation and Usage

### Prerequisites

- Python 3.7+
- Required packages: numpy, pandas, matplotlib, seaborn, qutip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantum-policy-hamiltonian.git
cd quantum-policy-hamiltonian

# Install dependencies
pip install numpy pandas matplotlib seaborn qutip
```

### Running the Model

```bash
# Generate synthetic data and run simulations
python quantum_simulation_runner.py
```

### Output Structure

The framework generates the following outputs:

- `synthetic_data/` - Contains CSV files with synthetic data
  - `wheat_policy_data.csv` - Policy data by country and year
  - `wheat_production_data.csv` - Production data by country and year
  - `wheat_trade_data.csv` - Bilateral trade flows
  - `wheat_stockpile_data.csv` - Stockpile data by country and year
  - `tariffs_coupling.csv` - Tariff coupling matrix
  - `subsidies_coupling.csv` - Subsidy coupling matrix
  - `stockpiling_coupling.csv` - Stockpile coupling matrix

- `quantum_results/` - Contains simulation results
  - `policy_evolution.csv` - Time evolution of policy levels
  - `equilibrium.csv` - Equilibrium policy levels
  - `coupling_matrices.png` - Visualization of coupling matrices
  - `policy_evolution.png` - Visualization of policy evolution
  - `equilibrium_policies.png` - Visualization of equilibrium policies
  - `scenario_comparison/` - Results from scenario analysis
  - `analysis/` - Additional analysis and metrics

## Interpreting Results

The key outputs to examine are:

1. **Policy Evolution** - Shows how policies evolve over time from initial conditions
2. **Equilibrium Policies** - Reveals stable policy configurations after evolution
3. **Coupling Matrices** - Illustrates the strength and nature of policy interactions
4. **Scenario Comparisons** - Demonstrates how different policy scenarios affect outcomes
5. **Stability Metrics** - Quantifies the volatility and stability of different policy regimes

## Extending the Model

The framework can be extended in several ways:

1. Add more countries or regions
2. Include additional policy dimensions
3. Incorporate real data from FAO, USDA, or other sources
4. Implement more sophisticated coupling mechanisms
5. Add economic impact calculations to assess welfare effects

## Mathematical Framework

The quantum policy Hamiltonian is formulated as:

```
H = H_local + H_coupling + H_transverse
```

Where:
- `H_local` represents current policy states and preferences
- `H_coupling` captures interactions between policies
- `H_transverse` introduces quantum fluctuations and uncertainty

Time evolution follows the Schrödinger equation:

```
|ψ(t)⟩ = e^(-iHt/ħ) |ψ(0)⟩
```

Policy expectations are calculated as:

```
⟨P_i⟩ = ⟨ψ(t)| P_i |ψ(t)⟩
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
