#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // Time series data
  DATA_VECTOR(Year);          // Year of observation
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m^2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  
  // Forcing data
  DATA_VECTOR(sst_dat);       // Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (individuals/m^2/year)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------

  // Coral growth parameters
  PARAMETER(r_fast);          // Intrinsic growth rate for fast-growing corals (year^-1)
  PARAMETER(r_slow);          // Intrinsic growth rate for slow-growing corals (year^-1)
  PARAMETER(K_coral);         // Total carrying capacity for all corals (% cover)
  
  // SST effects on coral growth
  PARAMETER(sst_opt_fast);    // Optimal SST for fast-growing coral growth (Celsius)
  PARAMETER(sst_tol_fast);    // SST tolerance for fast-growing corals (Celsius)
  PARAMETER(sst_opt_slow);    // Optimal SST for slow-growing coral growth (Celsius)
  PARAMETER(sst_tol_slow);    // SST tolerance for slow-growing corals (Celsius)

  // COTS predation parameters
  PARAMETER(alpha_cots);      // COTS attack rate (m^2/individual/year)
  PARAMETER(h_cots);          // COTS handling time per unit of coral (% cover^-1 * year)
  PARAMETER(pref_fast);       // COTS preference for fast-growing corals (dimensionless, 0-1)

  // COTS life history parameters
  PARAMETER(assimilation_eff); // Efficiency of converting consumed coral to COTS biomass (individuals/m^2 per % cover)
  PARAMETER(m_cots);           // COTS natural mortality rate (year^-1)
  PARAMETER(q_cots);           // COTS density-dependent mortality coefficient (m^2/individual/year)

  // Observation error parameters
  PARAMETER(log_sd_cots);     // Log of the standard deviation for COTS data
  PARAMETER(log_sd_fast);     // Log of the standard deviation for fast coral data
  PARAMETER(log_sd_slow);     // Log of the standard deviation for slow coral data

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------

  int n_steps = Year.size(); // Number of time steps in the simulation
  
  // Create vectors to store model predictions
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);

  // Initialize predictions with the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize the negative log-likelihood
  Type nll = 0.0;

  // Fixed minimum standard deviation for likelihood calculations to ensure stability
  Type min_sd = 0.01; 
  Type sd_cots = exp(log_sd_cots) + min_sd;
  Type sd_fast = exp(log_sd_fast) + min_sd;
  Type sd_slow = exp(log_sd_slow) + min_sd;

  // ------------------------------------------------------------------------
  // ECOLOGICAL PROCESS EQUATIONS
  // ------------------------------------------------------------------------
  //
  // 1. SST Effect (sst_effect): A Gaussian function determining the impact of temperature on coral growth.
  //    sst_effect = exp(-0.5 * ((sst - sst_opt) / sst_tol)^2)
  //
  // 2. Coral Growth (coral_growth): Logistic growth limited by total coral cover (K_coral).
  //    coral_growth = r * sst_effect * Coral * (1 - (TotalCoral / K_coral))
  //
  // 3. COTS Predation (predation): Holling Type II functional response with prey switching.
  //    predation_on_fast = COTS * (alpha * pref * FastCoral) / (1 + alpha * h * (pref * FastCoral + (1-pref) * SlowCoral))
  //    predation_on_slow = COTS * (alpha * (1-pref) * SlowCoral) / (1 + alpha * h * (pref * FastCoral + (1-pref) * SlowCoral))
  //
  // 4. COTS Growth (cots_growth): Growth from assimilated coral biomass.
  //    cots_growth = assimilation_eff * (predation_on_fast + predation_on_slow)
  //
  // 5. COTS Mortality (cots_mortality): Sum of natural and density-dependent mortality.
  //    cots_mortality = m * COTS + q * COTS^2
  //
  // 6. State Dynamics (Euler integration):
  //    Coral(t) = Coral(t-1) + coral_growth - predation
  //    COTS(t) = COTS(t-1) + cots_growth - cots_mortality + immigration
  //
  // ------------------------------------------------------------------------

  for (int i = 1; i < n_steps; ++i) {
    // --- Calculate intermediate terms at step i-1 ---
    
    // Total coral cover
    Type total_coral_prev = fast_pred(i-1) + slow_pred(i-1);
    
    // SST effect on coral growth
    Type sst_effect_fast = exp(Type(-0.5) * pow((sst_dat(i-1) - sst_opt_fast) / sst_tol_fast, 2));
    Type sst_effect_slow = exp(Type(-0.5) * pow((sst_dat(i-1) - sst_opt_slow) / sst_tol_slow, 2));

    // COTS functional response denominator
    Type fr_denom = Type(1.0) + alpha_cots * h_cots * (pref_fast * fast_pred(i-1) + (Type(1.0) - pref_fast) * slow_pred(i-1)) + Type(1e-8);

    // --- Calculate state changes ---

    // Fast-growing coral dynamics
    Type fast_growth = r_fast * sst_effect_fast * fast_pred(i-1) * (Type(1.0) - total_coral_prev / K_coral);
    Type predation_on_fast = cots_pred(i-1) * (alpha_cots * pref_fast * fast_pred(i-1)) / fr_denom;
    fast_pred(i) = fast_pred(i-1) + fast_growth - predation_on_fast;

    // Slow-growing coral dynamics
    Type slow_growth = r_slow * sst_effect_slow * slow_pred(i-1) * (Type(1.0) - total_coral_prev / K_coral);
    Type predation_on_slow = cots_pred(i-1) * (alpha_cots * (Type(1.0) - pref_fast) * slow_pred(i-1)) / fr_denom;
    slow_pred(i) = slow_pred(i-1) + slow_growth - predation_on_slow;

    // COTS dynamics
    Type cots_growth = assimilation_eff * (predation_on_fast + predation_on_slow);
    Type cots_mortality = m_cots * cots_pred(i-1) + q_cots * pow(cots_pred(i-1), 2);
    cots_pred(i) = cots_pred(i-1) + cots_growth - cots_mortality + cotsimm_dat(i-1);

    // --- Enforce non-negative predictions ---
    fast_pred(i) = CppAD::CondExpGe(fast_pred(i), Type(0.0), fast_pred(i), Type(1e-8));
    slow_pred(i) = CppAD::CondExpGe(slow_pred(i), Type(0.0), slow_pred(i), Type(1e-8));
    cots_pred(i) = CppAD::CondExpGe(cots_pred(i), Type(0.0), cots_pred(i), Type(1e-8));
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------

  for (int i = 0; i < n_steps; ++i) {
    // Lognormal distribution is used as abundances and cover are strictly positive
    nll -= dnorm(log(cots_dat(i)), log(cots_pred(i)), sd_cots, true);
    nll -= dnorm(log(fast_dat(i)), log(fast_pred(i)), sd_fast, true);
    nll -= dnorm(log(slow_dat(i)), log(slow_pred(i)), sd_slow, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
