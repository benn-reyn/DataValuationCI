import numpy as np
import pandas as pd
from pygam import LinearGAM, s
import matplotlib.pyplot as plt

# --- 0. INTEGRATED MODEL AND PARAMETERS ---

# Generic class for valuing a data-driven application in manufacturing
class DataAppValuation:
    def __init__(self):
        self.params = {
            # Base production parameters
            'planned_run_time': 7.5,        # hours per shift
            'design_rate': 240,             # units per hour
            'contribution_margin': 120,     # $ per good unit
            'unit_cogs': 80,                # cost of goods sold
            'downtime_cost_per_hr': 4500,   # $ per hour of downtime
            'carrying_cost_rate': 0.20,     # 20% annual carrying cost
            'energy_price': 0.12,           # $ per kWh
            'incident_cost': 50000,         # $ per safety incident
            
            # Baseline metrics (before application)
            'oee_base': 0.60,
            'defect_rate_base': 0.03,
            'downtime_hrs_base': 0.5,       # hours per shift
            'wip_base': 45,                 # days of WIP
            'incidents_base': 2,            # per quarter
            'kwh_base': 850,                # kWh per shift
            'cycle_time_std_base': 2.1,     # minutes std dev
            
            # Risk parameters
            'annual_profit': 2000000,       # Profit from this production line
            'historical_risk_costs': 75000, # Actual risk costs (expediting, overtime, etc.)
            'risk_appetite': 'moderate'
        }
        self.params['risk_lambda'] = self.calculate_realistic_lambda()
        self.params['shifts_per_year'] = 250 * 2 # 2 shifts/day

    def calculate_realistic_lambda(self):
        p = self.params
        if p['risk_appetite'] == 'conservative': base_lambda = 0.15
        elif p['risk_appetite'] == 'moderate': base_lambda = 0.08
        else: base_lambda = 0.03
        risk_cost_ratio = p['historical_risk_costs'] / p['annual_profit']
        adjusted_lambda = base_lambda * (1 + risk_cost_ratio * 5)
        return min(adjusted_lambda, 0.25)

    def calculate_total_annual_benefit(self, improvements):
        p = self.params
        benefits = {}
        
        # NOTE: Improvements are now relative (e.g., 0.1 = 10% improvement)
        
        # Throughput/OEE
        abs_oee_uplift = p['oee_base'] * improvements['oee_improvement']
        benefits['throughput'] = abs_oee_uplift * p['planned_run_time'] * \
                                 p['design_rate'] * p['contribution_margin'] * p['shifts_per_year']
        
        # Quality
        abs_defect_redux = p['defect_rate_base'] * improvements['quality_improvement']
        units_per_year = p['planned_run_time'] * p['design_rate'] * p['shifts_per_year']
        benefits['quality'] = abs_defect_redux * units_per_year * (p['unit_cogs'] + 20) # incl rework
        
        # Downtime
        abs_downtime_redux_hrs = p['downtime_hrs_base'] * improvements['downtime_reduction']
        benefits['downtime'] = abs_downtime_redux_hrs * p['downtime_cost_per_hr'] * p['shifts_per_year']
        
        # Inventory
        benefits['inventory'] = (p['wip_base'] * improvements['inventory_reduction']) * \
                                 p['carrying_cost_rate'] * 10000
        
        # Safety
        benefits['safety'] = (p['incidents_base'] * improvements['safety_improvement']) * \
                              p['incident_cost'] * 4
        
        # Energy
        benefits['energy'] = (p['kwh_base'] * improvements['energy_savings']) * \
                              p['energy_price'] * p['shifts_per_year']
        
        # Risk (Variability Reduction)
        std_new_sq = (p['cycle_time_std_base'] * (1 - improvements['variability_reduction']))**2
        profit_scale = p['annual_profit'] * 0.10
        benefits['risk'] = 0.5 * p['risk_lambda'] * \
                           (p['cycle_time_std_base']**2 - std_new_sq) * profit_scale
        
        return sum(benefits.values())

# --- EVSI Calculation Parameters ---
K = 20000 # Number of Monte Carlo simulations

# --- Uncertain Parameters (theta) ---
# Our prior beliefs about the *relative improvement* the new application will bring.
oee_improve_prior_mean = 0.10
oee_improve_prior_sd = 0.05
downtime_redux_prior_mean = 0.25
downtime_redux_prior_sd = 0.10
quality_improve_prior_mean = 0.30
quality_improve_prior_sd = 0.15
inventory_redux_prior_mean = 0.15
inventory_redux_prior_sd = 0.05
safety_improve_prior_mean = 0.40
safety_improve_prior_sd = 0.20
energy_redux_prior_mean = 0.08
energy_redux_prior_sd = 0.04
variability_redux_prior_mean = 0.20
variability_redux_prior_sd = 0.10

# --- Pilot Study Parameters ---
pilot_study_days = 30
pilot_study_daily_measurement_sd = 0.03 # Std dev of the daily ABSOLUTE OEE measurement

# --- 1. Step 1: Generate Probabilistic Sensitivity Analysis (PSA) Sample ---

# Instantiate the benefit model
valuation_model = DataAppValuation()
base_oee = valuation_model.params['oee_base']

print("Step 1: Generating PSA sample using the comprehensive benefit model...")

# Draw K samples from the prior distributions for ALL uncertain improvements
oee_improve_samples = np.random.normal(oee_improve_prior_mean, oee_improve_prior_sd, K)
downtime_redux_samples = np.random.normal(downtime_redux_prior_mean, downtime_redux_prior_sd, K)
quality_improve_samples = np.random.normal(quality_improve_prior_mean, quality_improve_prior_sd, K)
inventory_redux_samples = np.random.normal(inventory_redux_prior_mean, inventory_redux_prior_sd, K)
safety_improve_samples = np.random.normal(safety_improve_prior_mean, safety_improve_prior_sd, K)
energy_redux_samples = np.random.normal(energy_redux_prior_mean, energy_redux_prior_sd, K)
variability_redux_samples = np.random.normal(variability_redux_prior_mean, variability_redux_prior_sd, K)

# Calculate the total annual benefit for each of the K scenarios
total_annual_benefit = []
for i in range(K):
    improvements_dict = {
        'oee_improvement': oee_improve_samples[i],
        'downtime_reduction': downtime_redux_samples[i],
        'quality_improvement': quality_improve_samples[i],
        'inventory_reduction': inventory_redux_samples[i],
        'safety_improvement': safety_improve_samples[i],
        'energy_savings': energy_redux_samples[i],
        'variability_reduction': variability_redux_samples[i]
    }
    total_annual_benefit.append(valuation_model.calculate_total_annual_benefit(improvements_dict))

psa_df = pd.DataFrame({
    'oee_improve_sample': oee_improve_samples,
    'total_annual_benefit': total_annual_benefit
})
print(f"Generated {K} PSA samples.")

# --- 2. Step 2: Simulate Potential Data and Calculate Summary Statistics ---
print("\nStep 2: Simulating 30-day pilot project data...")
summary_statistics = []

for oee_relative_improve in psa_df['oee_improve_sample']:
    true_absolute_oee_uplift = oee_relative_improve * base_oee
    simulated_pilot_data = np.random.normal(loc=true_absolute_oee_uplift, 
                                            scale=pilot_study_daily_measurement_sd, 
                                            size=pilot_study_days)
    sample_mean = np.mean(simulated_pilot_data)
    summary_statistics.append(sample_mean)

psa_df['summary_statistic_oee'] = summary_statistics
print("Simulated pilot project for each PSA sample.")

# --- 3. Step 3: Fit the Nonparametric Regression (GAM) ---
print("\nStep 3: Fitting the Generalized Additive Model (GAM)...")
gam = LinearGAM(s(0, n_splines=15))
X = psa_df[['summary_statistic_oee']].values
y = psa_df['total_annual_benefit'].values
gam.fit(X, y)
psa_df['fitted_benefit_post_study'] = gam.predict(X)
print("GAM fitting complete.")

# --- 4. Step 4: Calculate EVSI ---
print("\nStep 4: Calculating EVSI...")

expected_benefit_with_info = np.mean(np.maximum(0, psa_df['fitted_benefit_post_study']))
benefit_without_info = np.maximum(0, np.mean(psa_df['fitted_benefit_post_study']))
evsi_annual = expected_benefit_with_info - benefit_without_info

print(f"\n--- Results (Based on Comprehensive Model) ---")
print(f"Expected Benefit with Pilot Info (Annual): ${expected_benefit_with_info:,.2f}")
print(f"Benefit of Acting on Current Info (Annual): ${benefit_without_info:,.2f}")
print(f"Total Annualized EVSI: ${evsi_annual:,.2f}")

# --- 5. Visualization ---
print("\nGenerating plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(psa_df['summary_statistic_oee']*100, psa_df['total_annual_benefit'],
           alpha=0.1, s=10, label='PSA Sample (Benefit based on Prior Beliefs)', color='lightcoral')

sorted_indices = np.argsort(psa_df['summary_statistic_oee'])
ax.plot(psa_df['summary_statistic_oee'].iloc[sorted_indices]*100,
        psa_df['fitted_benefit_post_study'].iloc[sorted_indices],
        color='darkred', lw=3, label='GAM Fit (Expected Benefit Post-Pilot)')

ax.set_xlabel("Pilot Project Result (Observed Absolute OEE Uplift in % points)", fontsize=12)
ax.set_ylabel("Total Annual Benefit ($)", fontsize=12)
ax.set_title("EVSI for a Data-Driven Application Pilot (Comprehensive Benefit Model)", fontsize=16)
ax.axhline(0, color='black', linestyle='--', lw=2, label='Decision Threshold (Benefit=0)')
ax.legend(fontsize=11)
plt.show()

# --- 6. Simple ROI Analysis ---
application_cost = 150000  # Cost for the application + integration
expected_benefit = benefit_without_info # Use the pre-pilot expected benefit for ROI
payback_period = application_cost / expected_benefit if expected_benefit > 0 else float('inf')

print(f"\n--- Simple ROI ANALYSIS ---")
print(f"Application Cost: ${application_cost:,.0f}")
print(f"Expected Annual Benefit (Pre-Pilot): ${expected_benefit:,.0f}")
if payback_period != float('inf'):
    print(f"Payback Period: {payback_period:.1f} years")
    print(f"1-Year ROI: {(expected_benefit - application_cost)/application_cost*100:.0f}%")
else:
    print("Payback cannot be calculated as the project is not expected to be profitable.")

