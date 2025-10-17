#include <TMB.hpp>

/*
================================================================================
* Model: Crown-of-Thorns Starfish (COTS) Outbreak Dynamics
================================================================================
*
* This TMB model simulates the population dynamics of COTS and their
* impact on fast-growing (Acropora) and slow-growing (Faviidae, Porites) corals.
*
* EQUATION DESCRIPTIONS:
*
* 1. COTS Predation (Multi-species Holling Type II Functional Response):
*    - Predation on each coral type depends on the coral's abundance, COTS attack rate,
*      handling time, and a preference factor. The rate saturates as coral becomes abundant.
*    - PredationRate_fast = (AttackRate * Preference_fast * FastCoral) / (1 + Denominator)
*    - PredationRate_slow = (AttackRate * (1-Preference_fast) * SlowCoral) / (1 + Denominator)
*    - Denominator = AttackRate * Preference_fast * HandlingTime * FastCoral +
*                    AttackRate * (1-Preference_fast) * HandlingTime * SlowCoral
*
* 2. Sea-Surface Temperature (SST) Effect:
*    - COTS feeding activity is modulated by SST via a Gaussian thermal performance curve.
*      Activity peaks at an optimal temperature (sst_opt_cots) and declines at temperatures
*      above or below this optimum.
*    - TempEffect = exp(-((SST - OptimalSST)^2) / (2 * NicheWidth^2))
*
* 3. COTS Population Dynamics (d_cots/dt):
*    - COTS biomass increases based on assimilated coral biomass (predation multiplied by
*      conversion efficiency) and external larval immigration.
*    - It decreases due to a constant natural mortality rate and a density-dependent
*      mortality rate that increases with the square of the COTS population.
*    - d_cots/dt = (Conversion_fast * Predation_fast + Conversion_slow * Predation_slow) * COTS
*                  - NaturalMortality * COTS - DensityDependentMortality * COTS^2
*                  + LarvalImmigration
*
* 4. Fast-Growing Coral Dynamics (d_fast/dt):
*    - Fast corals grow logistically, competing with slow corals for available space
*      (relative to a total carrying capacity, K_total).
*    - Population decreases due to predation by COTS.
*    - d_fast/dt = GrowthRate_fast * FastCoral * (1 - (FastCoral + Comp_fs * SlowCoral)/K_total)
*                  - PredationRate_fast * COTS
*
* 5. Slow-Growing Coral Dynamics (d_slow/dt):
*    - Slow corals also grow logistically, competing with fast corals for space.
*    - Population decreases due to predation by COTS.
*    - d_slow/dt = GrowthRate_slow * SlowCoral * (1 - (SlowCoral + Comp_sf * FastCoral)/K_total)
*                  - PredationRate_slow * COTS
*
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ==========================================================================
  // DATA INPUTS
  // ==========================================================================
  DATA_VECTOR(Year);          // Vector of years for the time series
  DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Forcing: Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // Forcing: COTS larval immigration rate (individuals/m2/year)

  // ==========================================================================
  // PARAMETERS
  // ==========================================================================
  // COTS Parameters
  PARAMETER(log_cots_ar);         // log(Attack Rate) of COTS on corals (m2/individual/year)
  PARAMETER(log_cots_h);          // log(Handling Time) per unit of coral consumed (year)
  PARAMETER(logit_cots_pref_fast);// logit(Preference) for fast-growing corals (0-1 scale)
  PARAMETER(log_cots_eff_fast);   // log(Conversion Efficiency) of fast coral to COTS biomass (dimensionless)
  PARAMETER(log_cots_eff_slow);   // log(Conversion Efficiency) of slow coral to COTS biomass (dimensionless)
  PARAMETER(log_cots_mort);       // log(Natural Mortality Rate) of COTS (year^-1)
  PARAMETER(log_cots_mort_dd);    // log(Density-Dependent Mortality Coefficient) for COTS ((individuals/m2)^-1 * year^-1)
  PARAMETER(sst_opt_cots);        // Optimal SST for COTS feeding (Celsius)
  PARAMETER(log_sst_width_cots);  // log(Niche Width) of SST tolerance for COTS feeding (Celsius)

  // Coral Parameters
  PARAMETER(log_fast_gr);         // log(Intrinsic Growth Rate) of fast-growing corals (year^-1)
  PARAMETER(log_slow_gr);         // log(Intrinsic Growth Rate) of slow-growing corals (year^-1)
  PARAMETER(log_K_total);         // log(Total Carrying Capacity) for all corals (% cover)
  PARAMETER(log_comp_fs);         // log(Competition Coefficient) of slow corals on fast corals (dimensionless)
  PARAMETER(log_comp_sf);         // log(Competition Coefficient) of fast corals on slow corals (dimensionless)

  // Observation Error Parameters
  PARAMETER(log_sd_cots);         // log(Standard Deviation) for COTS abundance observations
  PARAMETER(log_sd_fast);         // log(Standard Deviation) for fast coral cover observations
  PARAMETER(log_sd_slow);         // log(Standard Deviation) for slow coral cover observations

  // ==========================================================================
  // PARAMETER TRANSFORMATIONS
  // ==========================================================================
  // Apply transformations to ensure parameters are biologically meaningful (e.g., positive)
  Type cots_ar = exp(log_cots_ar);
  Type cots_h = exp(log_cots_h);
  Type cots_pref_fast = 1.0 / (1.0 + exp(-logit_cots_pref_fast));
  Type cots_eff_fast = exp(log_cots_eff_fast);
  Type cots_eff_slow = exp(log_cots_eff_slow);
  Type cots_mort = exp(log_cots_mort);
  Type cots_mort_dd = exp(log_cots_mort_dd);
  Type sst_width_cots = exp(log_sst_width_cots);
  Type fast_gr = exp(log_fast_gr);
  Type slow_gr = exp(log_slow_gr);
  Type K_total = exp(log_K_total);
  Type comp_fs = exp(log_comp_fs);
  Type comp_sf = exp(log_comp_sf);
  Type sd_cots = exp(log_sd_cots);
  Type sd_fast = exp(log_sd_fast);
  Type sd_slow = exp(log_sd_slow);

  // ==========================================================================
  // MODEL INITIALIZATION
  // ==========================================================================
  int n = Year.size(); // Number of time steps
  Type jnll = 0.0;     // Initialize joint negative log-likelihood

  // Prediction vectors for state variables
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial conditions from the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ==========================================================================
  // DYNAMIC MODEL (Time-stepping loop)
  // ==========================================================================
  for (int i = 1; i < n; ++i) {
    // --- Define state variables from previous time step ---
    Type COTS = cots_pred(i-1);
    Type FAST_CORAL = fast_pred(i-1);
    Type SLOW_CORAL = slow_pred(i-1);

    // --- Calculate SST effect on COTS feeding ---
    Type sst_effect = exp(-pow(sst_dat(i-1) - sst_opt_cots, 2) / (2.0 * pow(sst_width_cots, 2)));
    Type effective_ar = cots_ar * sst_effect;

    // --- Calculate COTS predation on corals (Multi-species Holling Type II) ---
    Type fr_denom = 1.0 + effective_ar * cots_pref_fast * cots_h * FAST_CORAL + effective_ar * (1.0 - cots_pref_fast) * cots_h * SLOW_CORAL + 1e-8;
    Type predation_on_fast = (effective_ar * cots_pref_fast * FAST_CORAL) / fr_denom;
    Type predation_on_slow = (effective_ar * (1.0 - cots_pref_fast) * SLOW_CORAL) / fr_denom;

    // --- Calculate change (derivatives) for each state variable ---
    Type d_cots = (cots_eff_fast * predation_on_fast + cots_eff_slow * predation_on_slow) * COTS - cots_mort * COTS - cots_mort_dd * pow(COTS, 2) + cotsimm_dat(i-1);
    Type d_fast = fast_gr * FAST_CORAL * (1.0 - (FAST_CORAL + comp_fs * SLOW_CORAL) / (K_total + 1e-8)) - predation_on_fast * COTS;
    Type d_slow = slow_gr * SLOW_CORAL * (1.0 - (SLOW_CORAL + comp_sf * FAST_CORAL) / (K_total + 1e-8)) - predation_on_slow * COTS;

    // --- Update predictions using Euler method (dt = 1 year) ---
    cots_pred(i) = COTS + d_cots;
    fast_pred(i) = FAST_CORAL + d_fast;
    slow_pred(i) = SLOW_CORAL + d_slow;

    // --- Enforce non-negativity constraint ---
    cots_pred(i) = (cots_pred(i) > Type(1e-8)) ? cots_pred(i) : Type(1e-8);
    fast_pred(i) = (fast_pred(i) > Type(1e-8)) ? fast_pred(i) : Type(1e-8);
    slow_pred(i) = (slow_pred(i) > Type(1e-8)) ? slow_pred(i) : Type(1e-8);
  }

  // ==========================================================================
  // LIKELIHOOD CALCULATION
  // ==========================================================================
  // Use a lognormal distribution for abundance/cover data, which must be positive.
  // This is equivalent to a normal distribution on the log-transformed data.
  // We sum the likelihood contributions from time step 1 onwards.
  for (int i = 1; i < n; ++i) {
    jnll -= dnorm(log(cots_dat(i) + 1e-8), log(cots_pred(i) + 1e-8), sd_cots, true);
    jnll -= dnorm(log(fast_dat(i) + 1e-8), log(fast_pred(i) + 1e-8), sd_fast, true);
    jnll -= dnorm(log(slow_dat(i) + 1e-8), log(slow_pred(i) + 1e-8), sd_slow, true);
  }

  // ==========================================================================
  // REPORTING SECTION
  // ==========================================================================
  // Report predicted time series for plotting and analysis
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Report standard deviations for diagnostics
  ADREPORT(sd_cots);
  ADREPORT(sd_fast);
  ADREPORT(sd_slow);

  return jnll;
}
