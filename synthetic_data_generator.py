#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:10:05 2025

   Generate synthetic wheat policy data for five major countries
   (USA, EU, Russia, China, India) for testing the quantum policy Hamiltonian.

@author: mjp38
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime

class SyntheticWheatPolicyData:
    
    def __init__(self, start_year=2018, end_year=2024, random_seed=42):
        """
        Initialize the synthetic data generator
        
        Parameters:
        -----------
        start_year : int
            First year for time series data
        end_year : int
            Last year for time series data
        random_seed : int
            Seed for random number generation (for reproducibility)
        """
        self.countries = ['USA', 'EU', 'Russia', 'China', 'India']
        self.years = list(range(start_year, end_year + 1))
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define baseline policy characteristics for each country
        self.policy_baselines = {
            'USA': {
                'import_tariff_rate': 3.0,          # Low wheat import tariffs
                'export_tax_rate': 0.0,             # No export taxes
                'direct_producer_subsidy': 40.0,    # Moderate producer subsidies
                'price_support_level': 175.0,       # Moderate price support
                'public_stockpile_target': 15.0,    # Moderate stockpile target
                'stock_to_use_ratio': 35.0,         # Moderate stock-to-use ratio
                'policy_intensity_index': 45.0      # Moderate policy intervention
            },
            'EU': {
                'import_tariff_rate': 60.0,         # Moderate-high import tariffs
                'export_tax_rate': 0.0,             # No export taxes
                'direct_producer_subsidy': 80.0,    # High producer subsidies through CAP
                'price_support_level': 200.0,       # Substantial price support
                'public_stockpile_target': 10.0,    # Moderate stockpile target
                'stock_to_use_ratio': 15.0,         # Modest stock-to-use ratio
                'policy_intensity_index': 65.0      # High policy intervention
            },
            'Russia': {
                'import_tariff_rate': 5.0,          # Low import tariffs
                'export_tax_rate': 10.0,            # Moderate export taxes/quotas
                'direct_producer_subsidy': 30.0,    # Moderate producer subsidies
                'price_support_level': 160.0,       # Moderate price support
                'public_stockpile_target': 18.0,    # Substantial stockpile target
                'stock_to_use_ratio': 14.0,         # Moderate stock-to-use ratio
                'policy_intensity_index': 70.0      # High policy intervention
            },
            'China': {
                'import_tariff_rate': 65.0,         # High import tariffs
                'export_tax_rate': 0.0,             # No export taxes (not a major exporter)
                'direct_producer_subsidy': 75.0,    # High producer subsidies
                'price_support_level': 380.0,       # Very high price support
                'public_stockpile_target': 135.0,   # Very high stockpile target
                'stock_to_use_ratio': 50.0,         # Very high stock-to-use ratio
                'policy_intensity_index': 90.0      # Very high policy intervention
            },
            'India': {
                'import_tariff_rate': 80.0,         # High import tariffs
                'export_tax_rate': 8.0,             # Some export restrictions
                'direct_producer_subsidy': 55.0,    # Substantial producer subsidies
                'price_support_level': 200.0,       # High price support
                'public_stockpile_target': 25.0,    # High stockpile target
                'stock_to_use_ratio': 15.0,         # Moderate stock-to-use ratio
                'policy_intensity_index': 75.0      # High policy intervention
            }
        }
        
        # Define production baselines
        self.production_baselines = {
            'USA': {
                'area_harvested': 16.0,            # Million hectares
                'yield': 3.3,                      # Tons/hectare
                'production_volume': 52.8,         # MMT
                'domestic_consumption': 30.0,      # MMT
                'feed_use_ratio': 0.1,             # 10% for feed
                'food_use_ratio': 0.8,             # 80% for food
                'exports': 23.0,                   # MMT
                'imports': 3.0,                    # MMT
                'stocks_to_use': 0.35              # 35% stocks-to-use
            },
            'EU': {
                'area_harvested': 25.0,            # Million hectares
                'yield': 5.4,                      # Tons/hectare
                'production_volume': 135.0,        # MMT
                'domestic_consumption': 110.0,     # MMT
                'feed_use_ratio': 0.4,             # 40% for feed
                'food_use_ratio': 0.53,            # 53% for food
                'exports': 30.0,                   # MMT
                'imports': 5.0,                    # MMT
                'stocks_to_use': 0.15              # 15% stocks-to-use 
            },
            'Russia': {
                'area_harvested': 28.0,            # Million hectares
                'yield': 3.0,                      # Tons/hectare
                'production_volume': 84.0,         # MMT
                'domestic_consumption': 42.0,      # MMT
                'feed_use_ratio': 0.4,             # 40% for feed
                'food_use_ratio': 0.55,            # 55% for food
                'exports': 45.0,                   # MMT
                'imports': 0.5,                    # MMT
                'stocks_to_use': 0.14              # 14% stocks-to-use 
            },
            'China': {
                'area_harvested': 24.0,            # Million hectares
                'yield': 5.8,                      # Tons/hectare
                'production_volume': 139.2,        # MMT
                'domestic_consumption': 150.0,     # MMT
                'feed_use_ratio': 0.23,            # 23% for feed
                'food_use_ratio': 0.7,             # 70% for food
                'exports': 0.5,                    # MMT
                'imports': 12.0,                   # MMT
                'stocks_to_use': 0.5               # 50% stocks-to-use 
            },
            'India': {
                'area_harvested': 30.0,            # Million hectares
                'yield': 3.5,                      # Tons/hectare
                'production_volume': 105.0,        # MMT
                'domestic_consumption': 100.0,     # MMT
                'feed_use_ratio': 0.06,            # 6% for feed
                'food_use_ratio': 0.9,             # 90% for food
                'exports': 5.0,                    # MMT
                'imports': 0.2,                    # MMT
                'stocks_to_use': 0.15              # 15% stocks-to-use
            }
        }
        
        # Define trade flow baselines (exporter -> importer)
        self.trade_flow_baselines = {
            # USA exports to
            ('USA', 'China'): 0.8,            # MMT
            ('USA', 'Japan'): 2.5,            # MMT (not in our 5 countries but important)
            ('USA', 'Mexico'): 3.0,           # MMT (not in our 5 countries but important)
            ('USA', 'Philippines'): 2.0,      # MMT (not in our 5 countries but important)
            ('USA', 'EU'): 0.3,               # MMT
            ('USA', 'India'): 0.1,            # MMT
            
            # EU exports to
            ('EU', 'China'): 1.0,             # MMT
            ('EU', 'USA'): 0.2,               # MMT
            ('EU', 'Egypt'): 5.0,             # MMT (not in our 5 countries but important)
            ('EU', 'Algeria'): 4.0,           # MMT (not in our 5 countries but important)
            ('EU', 'Morocco'): 3.8,           # MMT (not in our 5 countries but important)
            
            # Russia exports to
            ('Russia', 'Egypt'): 8.0,         # MMT (not in our 5 countries but important)
            ('Russia', 'Turkey'): 7.0,        # MMT (not in our 5 countries but important)
            ('Russia', 'China'): 0.5,         # MMT
            ('Russia', 'India'): 0.3,         # MMT
            ('Russia', 'EU'): 0.2,            # MMT
            
            # China exports to (minimal)
            ('China', 'India'): 0.05,         # MMT
            ('China', 'Russia'): 0.05,        # MMT
            
            # India exports to
            ('India', 'Bangladesh'): 2.0,     # MMT (not in our 5 countries but important)
            ('India', 'Nepal'): 0.5,          # MMT (not in our 5 countries but important)
            ('India', 'China'): 0.1,          # MMT
            ('India', 'EU'): 0.05,            # MMT
        }
        
        # Define policy events - discrete changes that occur in specific years
        self.policy_events = [
            # Format: (country, year, parameter, change_value, change_type)
            # change_type: 'absolute' or 'percentage'
            
            # Russia export restrictions in 2020-2021
            ('Russia', 2020, 'export_tax_rate', 15.0, 'absolute'),
            ('Russia', 2021, 'export_tax_rate', 25.0, 'absolute'),
            ('Russia', 2022, 'export_tax_rate', 15.0, 'absolute'),
            
            # China stockpiling increase 2022
            ('China', 2022, 'public_stockpile_target', 20.0, 'percentage'),
            ('China', 2022, 'stock_to_use_ratio', 15.0, 'percentage'),
            
            # India export restrictions 2022
            ('India', 2022, 'export_tax_rate', 20.0, 'absolute'),
            
            # EU tariff on Russian grain 2023
            ('EU', 2023, 'import_tariff_rate', 95.0, 'absolute'),
            
            # India tariff reduction 2024
            ('India', 2024, 'import_tariff_rate', -40.0, 'percentage'),
            
            # USA subsidy changes
            ('USA', 2023, 'direct_producer_subsidy', 10.0, 'percentage'),
        ]
        
        # Define production events - major changes in production
        self.production_events = [
            # Format: (country, year, parameter, change_value, change_type)
            
            # Russia bumper crop 2022
            ('Russia', 2022, 'production_volume', 25.0, 'percentage'),
            ('Russia', 2022, 'exports', 30.0, 'percentage'),
            
            # India drought effect 2023
            ('India', 2023, 'production_volume', -8.0, 'percentage'),
            ('India', 2023, 'imports', 400.0, 'percentage'),  # From very low base
            ('India', 2023, 'exports', -50.0, 'percentage'),
            
            # Russia drought 2024
            ('Russia', 2024, 'production_volume', -12.0, 'percentage'),
            ('Russia', 2024, 'exports', -8.0, 'percentage'),
            
            # EU reduced planting 2024
            ('EU', 2024, 'area_harvested', -4.0, 'percentage'),
            ('EU', 2024, 'production_volume', -4.0, 'percentage'),
        ]
        
        # Generate empty dataframes to be filled
        self.policy_data = None
        self.production_data = None
        self.trade_data = None
        self.stockpile_data = None
    
    def generate_policy_data(self):
        """
        Generate synthetic policy data with gradual trends and policy events
        
        Returns:
        --------
        pandas.DataFrame
            Policy data with countries and years as multi-index
        """
        # Create multi-index for policy dataframe
        index = pd.MultiIndex.from_product([self.countries, self.years], names=['Country', 'Year'])
        
        # Policy dimensions to track
        columns = [
            'import_tariff_rate',
            'export_tax_rate',
            'direct_producer_subsidy',
            'price_support_level',
            'public_stockpile_target',
            'stock_to_use_ratio',
            'policy_intensity_index'
        ]
        
        # Create empty dataframe
        policy_df = pd.DataFrame(index=index, columns=columns)
        
        # Fill with baseline data and add gradual trends
        for country in self.countries:
            baseline = self.policy_baselines[country]
            
            for year_idx, year in enumerate(self.years):
                for param, base_value in baseline.items():
                    if param in columns:
                        # Add small random variation + slight trend over time
                        trend_factor = 1.0 + 0.005 * year_idx  # 0.5% increase per year
                        random_factor = np.random.normal(1.0, 0.03)  # 3% random noise
                        
                        # Combine baseline, trend and randomness
                        value = base_value * trend_factor * random_factor
                        policy_df.loc[(country, year), param] = value
        
        # Apply policy events (abrupt changes)
        for country, year, param, change, change_type in self.policy_events:
            if (country, year) in policy_df.index and param in policy_df.columns:
                current_value = policy_df.loc[(country, year), param]
                
                if change_type == 'absolute':
                    new_value = current_value + change
                else:  # percentage change
                    new_value = current_value * (1 + change/100)
                    
                # Update the value for this year
                policy_df.loc[(country, year), param] = new_value
                
                # Propagate the change to future years (with some reversion to trend)
                for future_year in [y for y in self.years if y > year]:
                    current_value = policy_df.loc[(country, future_year), param]
                    years_since_event = future_year - year
                    
                    if change_type == 'absolute':
                        # Revert back towards trend by 30% per year
                        remaining_effect = change * (0.7 ** years_since_event)
                        policy_df.loc[(country, future_year), param] = current_value + remaining_effect
                    else:  # percentage change
                        # Revert back towards trend by 30% per year
                        remaining_effect = change * (0.7 ** years_since_event)
                        factor = 1 + remaining_effect/100
                        policy_df.loc[(country, future_year), param] = current_value * factor
        
        # Store the result
        self.policy_data = policy_df
        
        return policy_df
    
    def generate_production_data(self):
        """
        Generate synthetic production data with trends and events
        
        Returns:
        --------
        pandas.DataFrame
            Production data with countries and years as multi-index
        """
        # Create multi-index for production dataframe
        index = pd.MultiIndex.from_product([self.countries, self.years], names=['Country', 'Year'])
        
        # Production dimensions to track
        columns = [
            'area_harvested',
            'yield',
            'production_volume',
            'domestic_consumption',
            'feed_use',
            'food_use',
            'exports',
            'imports',
            'beginning_stocks',
            'ending_stocks',
            'stock_to_use_ratio'
        ]
        
        # Create empty dataframe
        production_df = pd.DataFrame(index=index, columns=columns)
        
        # First pass: generate base production and consumption data
        for country in self.countries:
            baseline = self.production_baselines[country]
            
            # Pre-calculate beginning stocks for first year
            beginning_stocks_first_year = baseline['domestic_consumption'] * baseline['stocks_to_use']
            
            for year_idx, year in enumerate(self.years):
                # Base random factors
                area_random = np.random.normal(1.0, 0.02)  # 2% variation
                yield_random = np.random.normal(1.0, 0.04)  # 4% variation (more weather dependent)
                consumption_random = np.random.normal(1.0, 0.02)  # 2% variation
                
                # Apply trends (small increases in productivity and consumption)
                area_trend = 1.0 + 0.001 * year_idx  # 0.1% increase per year
                yield_trend = 1.0 + 0.005 * year_idx  # 0.5% increase per year
                consumption_trend = 1.0 + 0.010 * year_idx  # 1.0% increase per year
                
                # Calculate core values
                area = baseline['area_harvested'] * area_trend * area_random
                crop_yield = baseline['yield'] * yield_trend * yield_random
                production = area * crop_yield
                
                consumption = baseline['domestic_consumption'] * consumption_trend * consumption_random
                feed_use = consumption * baseline['feed_use_ratio']
                food_use = consumption * baseline['food_use_ratio']
                
                # Handle stocks for first year specially
                if year_idx == 0:
                    beginning_stocks = beginning_stocks_first_year
                else:
                    # For later years, beginning stocks = ending stocks of previous year
                    prev_year = self.years[year_idx - 1]
                    beginning_stocks = production_df.loc[(country, prev_year), 'ending_stocks']
                
                # Calculate trade (with randomness)
                export_random = np.random.normal(1.0, 0.05)  # 5% variation
                import_random = np.random.normal(1.0, 0.05)  # 5% variation
                
                exports = baseline['exports'] * export_random
                imports = baseline['imports'] * import_random
                
                # Calculate ending stocks (balance equation)
                ending_stocks = beginning_stocks + production + imports - consumption - exports
                stock_to_use = ending_stocks / consumption
                
                # Store values
                production_df.loc[(country, year), 'area_harvested'] = area
                production_df.loc[(country, year), 'yield'] = crop_yield
                production_df.loc[(country, year), 'production_volume'] = production
                production_df.loc[(country, year), 'domestic_consumption'] = consumption
                production_df.loc[(country, year), 'feed_use'] = feed_use
                production_df.loc[(country, year), 'food_use'] = food_use
                production_df.loc[(country, year), 'exports'] = exports
                production_df.loc[(country, year), 'imports'] = imports
                production_df.loc[(country, year), 'beginning_stocks'] = beginning_stocks
                production_df.loc[(country, year), 'ending_stocks'] = ending_stocks
                production_df.loc[(country, year), 'stock_to_use_ratio'] = stock_to_use * 100  # as percentage
        
        # Apply production events
        for country, year, param, change, change_type in self.production_events:
            if (country, year) in production_df.index and param in production_df.columns:
                current_value = production_df.loc[(country, year), param]
                
                if change_type == 'absolute':
                    new_value = current_value + change
                else:  # percentage change
                    new_value = current_value * (1 + change/100)
                    
                # Update the value for this year
                production_df.loc[(country, year), param] = new_value
                
                # If production volume changes, update ending stocks
                if param == 'production_volume':
                    # Recalculate ending stocks
                    country_year = (country, year)
                    beginning = production_df.loc[country_year, 'beginning_stocks']
                    production = production_df.loc[country_year, 'production_volume']
                    imports = production_df.loc[country_year, 'imports']
                    consumption = production_df.loc[country_year, 'domestic_consumption']
                    exports = production_df.loc[country_year, 'exports']
                    
                    ending_stocks = beginning + production + imports - consumption - exports
                    production_df.loc[country_year, 'ending_stocks'] = ending_stocks
                    production_df.loc[country_year, 'stock_to_use_ratio'] = (ending_stocks / consumption) * 100
                
                # Update beginning stocks for the following year
                if year_idx + 1 < len(self.years) and param in ['production_volume', 'exports', 'imports']:
                    next_year = self.years[year_idx + 1]
                    production_df.loc[(country, next_year), 'beginning_stocks'] = production_df.loc[(country, year), 'ending_stocks']
        
        # Store the result
        self.production_data = production_df
        
        return production_df
    
    def generate_trade_data(self):
        """
        Generate synthetic bilateral trade data
        
        Returns:
        --------
        pandas.DataFrame
            Trade flow data with exporter, importer, and years as multi-index
        """
        # Create index for all possible trade flows and years
        exporters = self.countries
        importers = self.countries
        flow_pairs = [(ex, im) for ex in exporters for im in importers if ex != im]
        
        index = pd.MultiIndex.from_product([flow_pairs, self.years], names=['Trade_Pair', 'Year'])
        index = pd.MultiIndex.from_tuples([(ex, im, yr) for (ex, im), yr in index], 
                                          names=['Exporter', 'Importer', 'Year'])
        
        # Columns for trade data
        columns = [
            'wheat_volume',          # Trade volume in MMT
            'wheat_value',           # Trade value in USD millions
            'average_price'          # Average price in USD/ton
        ]
        
        # Create empty dataframe
        trade_df = pd.DataFrame(index=index, columns=columns)
        
        # Generate base trade flow data
        for (exporter, importer), base_volume in self.trade_flow_baselines.items():
            for year_idx, year in enumerate(self.years):
                if (exporter, importer, year) in trade_df.index:
                    # Add random variation and trend
                    trend_factor = 1.0 + 0.01 * year_idx  # 1% increase per year
                    random_factor = np.random.normal(1.0, 0.08)  # 8% random variation
                    
                    # Generate volume
                    volume = base_volume * trend_factor * random_factor
                    
                    # Generate price (global wheat price with variation)
                    # Base wheat price around $250/ton with upward trend and variation
                    base_price = 250 * (1 + 0.02 * year_idx)  # 2% price inflation per year
                    price_random = np.random.normal(1.0, 0.07)  # 7% price variation
                    price = base_price * price_random
                    
                    # Calculate value
                    value = volume * price / 1000  # Convert to USD millions
                    
                    # Store values
                    trade_df.loc[(exporter, importer, year), 'wheat_volume'] = volume
                    trade_df.loc[(exporter, importer, year), 'wheat_value'] = value
                    trade_df.loc[(exporter, importer, year), 'average_price'] = price
        
        # Modify trade flows based on policy events
        # For example, export restrictions reduce exports
        for country, year, param, change, change_type in self.policy_events:
            # If export tax increases, reduce exports from that country
            if param == 'export_tax_rate' and change > 0:
                for importer in importers:
                    if (country, importer, year) in trade_df.index:
                        current_volume = trade_df.loc[(country, importer, year), 'wheat_volume'] 
                        
                        if not pd.isna(current_volume) and current_volume > 0:
                            # Higher export tax reduces volume
                            reduction_factor = 0.95  # 5% reduction per 10% tax
                            
                            if change_type == 'absolute':
                                # Adjust based on absolute change
                                reduction = reduction_factor ** (change / 10)
                            else:
                                # Adjust based on percentage change
                                reduction = reduction_factor ** (change / 1000)
                                
                            new_volume = current_volume * reduction
                            
                            # Update volume and value
                            price = trade_df.loc[(country, importer, year), 'average_price']
                            trade_df.loc[(country, importer, year), 'wheat_volume'] = new_volume
                            trade_df.loc[(country, importer, year), 'wheat_value'] = new_volume * price / 1000
            
            # If import tariff increases, reduce imports to that country
            elif param == 'import_tariff_rate' and change > 0:
                for exporter in exporters:
                    if (exporter, country, year) in trade_df.index:
                        current_volume = trade_df.loc[(exporter, country, year), 'wheat_volume']
                        
                        if not pd.isna(current_volume) and current_volume > 0:
                            # Higher import tariff reduces volume
                            reduction_factor = 0.93  # 7% reduction per 10% tariff
                            
                            if change_type == 'absolute':
                                # Adjust based on absolute change
                                reduction = reduction_factor ** (change / 10)
                            else:
                                # Adjust based on percentage change
                                reduction = reduction_factor ** (change / 1000)
                                
                            new_volume = current_volume * reduction
                            
                            # Update volume and value
                            price = trade_df.loc[(exporter, country, year), 'average_price']
                            trade_df.loc[(exporter, country, year), 'wheat_volume'] = new_volume
                            trade_df.loc[(exporter, country, year), 'wheat_value'] = new_volume * price / 1000
        
        # Also modify trade based on production events
        for country, year, param, change, change_type in self.production_events:
            # If production decreases, reduce exports
            if param == 'production_volume' and change < 0:
                for importer in importers:
                    if (country, importer, year) in trade_df.index:
                        current_volume = trade_df.loc[(country, importer, year), 'wheat_volume']
                        
                        if not pd.isna(current_volume) and current_volume > 0:
                            # Production reduction reduces exports proportionally
                            if change_type == 'percentage':
                                # Convert to positive number for reduction factor
                                reduction = 1 + change/100  # e.g., -10% becomes 0.9
                                new_volume = current_volume * reduction
                                
                                # Update volume and value
                                price = trade_df.loc[(country, importer, year), 'average_price']
                                # Reduced supply increases price
                                price_factor = 1 - change/200  # e.g., -10% production increases price by ~5%
                                new_price = price * price_factor
                                
                                trade_df.loc[(country, importer, year), 'wheat_volume'] = new_volume
                                trade_df.loc[(country, importer, year), 'average_price'] = new_price
                                trade_df.loc[(country, importer, year), 'wheat_value'] = new_volume * new_price / 1000
        
        # Drop rows with zero or NaN trade
        trade_df = trade_df.dropna(subset=['wheat_volume'])
        trade_df = trade_df[trade_df['wheat_volume'] > 0.001]  # Minimum 1,000 tons
        
        # Store the result
        self.trade_data = trade_df
        
        return trade_df
    
    def generate_stockpile_data(self):
        """
        Generate synthetic stockpile policy data
        
        Returns:
        --------
        pandas.DataFrame
            Detailed stockpile data with countries and years as multi-index
        """
        # Create multi-index for stockpile dataframe
        index = pd.MultiIndex.from_product([self.countries, self.years], names=['Country', 'Year'])
        
        # Stockpile policy details to track
        columns = [
            'total_wheat_stocks',              # Total wheat stocks (MMT)
            'public_wheat_stocks',             # Public wheat stocks (MMT)
            'private_wheat_stocks',            # Private wheat stocks (MMT)
            'target_stock_level',              # Target stock level (MMT)
            'annual_stockpile_change',         # Annual change in stockpile (MMT)
            'stockpile_acquisition_price',     # Average acquisition price (USD/ton)
            'stockpile_release_price',         # Average release price (USD/ton)
            'storage_cost',                    # Annual storage cost (USD millions)
            'storage_loss_rate',               # Annual loss rate (%)
            'stocks_to_domestic_use',          # Stocks to domestic use ratio (%)
            'months_of_consumption',           # Months of consumption in storage
            'price_stabilization_effect'       # Estimated price stabilization effect (%)
        ]
        
        # Create empty dataframe
        stockpile_df = pd.DataFrame(index=index, columns=columns)
        
        # Public stock ratios by country
        public_stock_ratios = {
            'USA': 0.55,    # 55% public, 45% private
            'EU': 0.40,     # 40% public, 60% private
            'Russia': 0.70, # 70% public, 30% private
            'China': 0.88,  # 88% public, 12% private
            'India': 0.75   # 75% public, 25% private
        }
        
        # Fill stockpile data based on production data
        for country in self.countries:
            for year_idx, year in enumerate(self.years):
                if self.production_data is not None and (country, year) in self.production_data.index:
                    prod_data = self.production_data.loc[(country, year)]
                    
                    # Total stocks from production data
                    total_stocks = prod_data['ending_stocks']
                    stockpile_df.loc[(country, year), 'total_wheat_stocks'] = total_stocks
                    
                    # Public/private split
                    public_ratio = public_stock_ratios[country]
                    stockpile_df.loc[(country, year), 'public_wheat_stocks'] = total_stocks * public_ratio
                    stockpile_df.loc[(country, year), 'private_wheat_stocks'] = total_stocks * (1 - public_ratio)
                    
                    # Target stock level from policy data
                    if self.policy_data is not None and (country, year) in self.policy_data.index:
                        policy = self.policy_data.loc[(country, year)]
                        stockpile_df.loc[(country, year), 'target_stock_level'] = policy['public_stockpile_target']
                    
                    # Annual change in stockpile
                    if year_idx > 0:
                        prev_year = self.years[year_idx - 1]
                        prev_stocks = stockpile_df.loc[(country, prev_year), 'total_wheat_stocks']
                        annual_change = total_stocks - prev_stocks
                        stockpile_df.loc[(country, year), 'annual_stockpile_change'] = annual_change
                    
                    # Acquisition and release prices
                    # Base wheat price around $250/ton with upward trend and variation
                    base_price = 250 * (1 + 0.02 * year_idx)  # 2% inflation per year
                    
                    # Acquisition price is typically 5-10% above market price
                    acq_premium = np.random.uniform(1.05, 1.10)
                    stockpile_df.loc[(country, year), 'stockpile_acquisition_price'] = base_price * acq_premium
                    
                    # Release price varies by country (some subsidize, some profit)
                    if country in ['USA', 'EU']:
                        # Market-oriented, small markup
                        rel_factor = np.random.uniform(1.02, 1.05)
                    elif country in ['China', 'India']:
                        # May subsidize release price
                        rel_factor = np.random.uniform(0.90, 1.0)
                    else:
                        # Moderate markup
                        rel_factor = np.random.uniform(1.0, 1.05)
                    
                    stockpile_df.loc[(country, year), 'stockpile_release_price'] = base_price * rel_factor
                    
                    # Storage costs (depends on country infrastructure)
                    if country in ['USA', 'EU']:
                        # More efficient storage
                        unit_cost = np.random.uniform(25, 35)  # USD/ton/year
                    elif country in ['China', 'Russia']:
                        # Moderate efficiency
                        unit_cost = np.random.uniform(30, 45)  # USD/ton/year
                    else:
                        # Less efficient
                        unit_cost = np.random.uniform(40, 55)  # USD/ton/year
                    
                    storage_cost = total_stocks * unit_cost / 1000  # USD millions
                    stockpile_df.loc[(country, year), 'storage_cost'] = storage_cost
                    
                    # Storage loss rate (depends on country infrastructure)
                    if country in ['USA', 'EU']:
                        # More efficient storage
                        loss_rate = np.random.uniform(0.5, 1.2)  # %
                    elif country in ['China', 'Russia']:
                        # Moderate efficiency
                        loss_rate = np.random.uniform(1.0, 2.5)  # %
                    else:
                        # Less efficient
                        loss_rate = np.random.uniform(2.0, 4.5)  # %
                    
                    stockpile_df.loc[(country, year), 'storage_loss_rate'] = loss_rate
                    
                    # Stocks to domestic use and months of consumption
                    domestic_use = prod_data['domestic_consumption']
                    stocks_to_use = (total_stocks / domestic_use) * 100  # as percentage
                    months_consumption = (total_stocks / domestic_use) * 12  # months
                    
                    stockpile_df.loc[(country, year), 'stocks_to_domestic_use'] = stocks_to_use
                    stockpile_df.loc[(country, year), 'months_of_consumption'] = months_consumption
                    
                    # Price stabilization effect (rough estimate based on stock level)
                    # Higher stocks relative to target = higher stabilization
                    if 'target_stock_level' in stockpile_df.columns:
                        target = stockpile_df.loc[(country, year), 'target_stock_level']
                        if not pd.isna(target) and target > 0:
                            ratio = public_ratio * total_stocks / target
                            # Diminishing returns on stabilization
                            stabilization = min(50, 30 * ratio**0.5)  # Cap at 50%
                            stockpile_df.loc[(country, year), 'price_stabilization_effect'] = stabilization
        
        # Store the result
        self.stockpile_data = stockpile_df
        
        return stockpile_df
    
    def generate_all_data(self):
        """
        Generate all synthetic data types at once
        
        Returns:
        --------
        dict
            Dictionary containing all generated dataframes
        """
        # Generate in the correct order (some depend on others)
        policy_df = self.generate_policy_data()
        production_df = self.generate_production_data()
        trade_df = self.generate_trade_data()
        stockpile_df = self.generate_stockpile_data()
        
        return {
            'policy_data': policy_df,
            'production_data': production_df,
            'trade_data': trade_df,
            'stockpile_data': stockpile_df
        }
    
    def export_to_csv(self, output_dir="."):
        """
        Export all dataframes to CSV files
        
        Parameters:
        -----------
        output_dir : str
            Directory to save CSV files
        """
        import os
        
        # Ensure directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Check if data has been generated
        if self.policy_data is None:
            self.generate_all_data()
        
        # Export each dataframe
        self.policy_data.to_csv(f"{output_dir}/wheat_policy_data.csv")
        self.production_data.to_csv(f"{output_dir}/wheat_production_data.csv")
        self.trade_data.to_csv(f"{output_dir}/wheat_trade_data.csv")
        self.stockpile_data.to_csv(f"{output_dir}/wheat_stockpile_data.csv")
        
        print(f"Data exported to {output_dir}/")
    
    def calculate_policy_coupling_matrices(self):
        """
        Calculate policy coupling matrices for the Hamiltonian
        
        Returns:
        --------
        dict
            Dictionary containing coupling matrices for tariffs, subsidies, and stockpiling
        """
        # Check if data has been generated
        if self.policy_data is None or self.trade_data is None:
            self.generate_all_data()
        
        # Get latest year data
        latest_year = max(self.years)
        
        # Initialize coupling matrices
        tariff_coupling = pd.DataFrame(0.0, index=self.countries, columns=self.countries)
        subsidy_coupling = pd.DataFrame(0.0, index=self.countries, columns=self.countries)
        stockpile_coupling = pd.DataFrame(0.0, index=self.countries, columns=self.countries)
        
        # Set diagonal (self-coupling) to 1.0
        for country in self.countries:
            tariff_coupling.loc[country, country] = 1.0
            subsidy_coupling.loc[country, country] = 1.0
            stockpile_coupling.loc[country, country] = 1.0
        
        # Calculate off-diagonal coupling based on:
        # 1. Trade relationships (stronger trade = stronger coupling)
        # 2. Policy similarity (similar policies = positive coupling, different policies = negative coupling)
        # 3. Economic size (larger countries have stronger influence)
        
        # Get bilateral trade volumes
        bilateral_trade = {}
        for exporter in self.countries:
            for importer in self.countries:
                if exporter != importer:
                    # Find all trade flows from exporter to importer in latest year
                    trade_flows = self.trade_data[
                        (self.trade_data.index.get_level_values('Exporter') == exporter) &
                        (self.trade_data.index.get_level_values('Importer') == importer) &
                        (self.trade_data.index.get_level_values('Year') == latest_year)
                    ]
                    
                    # Sum volumes if multiple flows exist
                    total_volume = trade_flows['wheat_volume'].sum() if not trade_flows.empty else 0
                    bilateral_trade[(exporter, importer)] = total_volume
        
        # Get production sizes for economic weight
        production_sizes = {}
        for country in self.countries:
            country_prod = self.production_data.loc[(country, latest_year), 'production_volume']
            production_sizes[country] = country_prod
        
        # Get policy values
        tariff_rates = {}
        subsidy_levels = {}
        stockpile_ratios = {}
        
        for country in self.countries:
            if (country, latest_year) in self.policy_data.index:
                policy = self.policy_data.loc[(country, latest_year)]
                tariff_rates[country] = policy['import_tariff_rate']
                subsidy_levels[country] = policy['direct_producer_subsidy']
                stockpile_ratios[country] = policy['stock_to_use_ratio']
        
        # Calculate coupling strengths
        for country1 in self.countries:
            for country2 in self.countries:
                if country1 != country2:
                    # Get bilateral trade (both directions)
                    trade_12 = bilateral_trade.get((country1, country2), 0)
                    trade_21 = bilateral_trade.get((country2, country1), 0)
                    total_trade = trade_12 + trade_21
                    
                    # Normalize trade to [0, 1] scale
                    # Assuming 10 MMT is very high trade volume
                    trade_factor = min(1.0, total_trade / 10.0)
                    
                    # Economic size factor (larger countries have more influence)
                    size1 = production_sizes.get(country1, 0)
                    size2 = production_sizes.get(country2, 0)
                    size_factor = (size1 + size2) / 200  # Normalize assuming 200 MMT is very large
                    size_factor = min(1.0, size_factor)
                    
                    # Calculate tariff coupling
                    if country1 in tariff_rates and country2 in tariff_rates:
                        # Policy similarity for tariffs
                        tariff1 = tariff_rates[country1]
                        tariff2 = tariff_rates[country2]
                        tariff_diff = abs(tariff1 - tariff2) / 100  # Normalize by assuming 100 percentage points is maximum difference
                        
                        # Higher difference = more negative coupling
                        tariff_similarity = 1 - tariff_diff
                        
                        # Combine factors
                        # Trade partners with similar policies: positive coupling
                        # Trade partners with different policies: negative coupling
                        tariff_strength = (2 * tariff_similarity - 1) * trade_factor * size_factor
                        tariff_coupling.loc[country1, country2] = tariff_strength * 0.5  # Scale to reasonable range
                    
                    # Calculate subsidy coupling
                    if country1 in subsidy_levels and country2 in subsidy_levels:
                        # Policy similarity for subsidies
                        subsidy1 = subsidy_levels[country1]
                        subsidy2 = subsidy_levels[country2]
                        subsidy_diff = abs(subsidy1 - subsidy2) / 100  # Normalize by assuming 100 USD/ton is maximum difference
                        
                        # Higher difference = more negative coupling
                        subsidy_similarity = 1 - subsidy_diff
                        
                        # Combine factors
                        subsidy_strength = (2 * subsidy_similarity - 1) * trade_factor * size_factor
                        subsidy_coupling.loc[country1, country2] = subsidy_strength * 0.5
                    
                    # Calculate stockpile coupling
                    if country1 in stockpile_ratios and country2 in stockpile_ratios:
                        # Policy similarity for stockpiling
                        stock1 = stockpile_ratios[country1]
                        stock2 = stockpile_ratios[country2]
                        stock_diff = abs(stock1 - stock2) / 50  # Normalize by assuming 50 percentage points is maximum difference
                        
                        # Higher difference = more negative coupling
                        stock_similarity = 1 - stock_diff
                        
                        # Combine factors
                        stock_strength = (2 * stock_similarity - 1) * trade_factor * size_factor
                        stockpile_coupling.loc[country1, country2] = stock_strength * 0.5
        
        return {
            'tariffs': tariff_coupling,
            'subsidies': subsidy_coupling,
            'stockpiling': stockpile_coupling
        }


def main():
    """
    Main function to demonstrate synthetic data generation
    """
    # Create data generator
    generator = SyntheticWheatPolicyData(start_year=2018, end_year=2024)
    
    # Generate all data
    data = generator.generate_all_data()
    
    # Export to CSV
    generator.export_to_csv("synthetic_data")
    
    # Calculate coupling matrices
    coupling_matrices = generator.calculate_policy_coupling_matrices()
    
    # Export coupling matrices
    for name, matrix in coupling_matrices.items():
        matrix.to_csv(f"synthetic_data/{name}_coupling.csv")
    
    print("Synthetic data generation complete!")
    print("Files saved to synthetic_data/ directory")


if __name__ == "__main__":
    main()