#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DATA
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  // Forcing data vectors
  DATA_VECTOR(sst_dat);       // Sea-Surface Temperature in Celsius
  DATA_VECTOR(cotsimm_dat);   // Crown-of-thorns larval immigration rate in individuals/m2/year

  // Response data vectors
  DATA_VECTOR(cots_dat);      // Adult Class Crown-of-thorns starfish abundance in individuals/m2
  DATA_VECTOR(fast_dat);      // Fast-growing coral (Acropora spp.) cover in %
  DATA_VECTOR(slow_dat);      // Slow-growing coral (Faviidae spp. and Porities spp.) cover in %

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // PARAMETERS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // COTS parameters
  PARAMETER(assim_eff);      // COTS assimilation efficiency (dimensionless)
  PARAMETER(m_cots);         // COTS natural mortality rate (year^-1)
  PARAMETER(pred_rate_fast); // Max predation rate on fast corals (% cover eaten per COTS per year)
  PARAMETER(h_fast);         // Half-saturation constant for predation on fast corals (% cover)
  PARAMETER(pred_rate_slow); // Max predation rate on slow corals (% cover eaten per COTS per year)
  PARAMETER(h_slow);         // Half-saturation constant for predation on slow corals (% cover)

  // Coral parameters
  PARAMETER(r_fast);         // Intrinsic growth rate of fast corals (year^-1)
  PARAMETER(T_opt_fast);     // Optimal temperature for fast coral growth (Celsius)
  PARAMETER(T_tol_fast);     // Temperature tolerance for fast coral growth (Celsius)
  PARAMETER(m_fast);         // Natural mortality rate of fast corals (year^-1)
  PARAMETER(r_slow);         // Intrinsic growth rate of slow corals (year^-1)
  PARAMETER(T_opt_slow);     // Optimal temperature for slow coral growth (Celsius)
  PARAMETER(T_tol_slow);     // Temperature tolerance for slow coral growth (Celsius)
  PARAMETER(m_slow);         // Natural mortality rate of slow corals (year^-1)
  PARAMETER(K);              // Total coral carrying capacity (% cover)

  // Observation error parameters
  PARAMETER(log_sd_cots);    // Log of standard deviation for COTS abundance
  PARAMETER(log_sd_fast);    // Log of standard deviation for fast coral cover
  PARAMETER(log_sd_slow);    // Log of standard deviation for slow coral cover

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // INITIALIZATION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Transform SDs from log space to ensure they are positive
  Type sd_cots = exp(log_sd_cots);
  Type sd_fast = exp(log_sd_fast);
  Type sd_slow = exp(log_sd_slow);

  int n_steps = cots_dat.size(); // Number of time steps in the data
  
  // Prediction vectors
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);

  // Initialize predictions with the first data point from the observed data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize negative log-likelihood
  Type nll = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // EQUATIONS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 1. COTS Predation (Holling Type II functional response): 
  //    pred_on_fast = pred_rate_fast * fast_cover / (h_fast + fast_cover)
  //    pred_on_slow = pred_rate_slow * slow_cover / (h_slow + slow_cover)
  // 2. COTS Population Dynamics:
  //    cots(t) = cots(t-1) * (1 - m_cots) + cots(t-1) * assim_eff * (pred_on_fast + pred_on_slow) + cots_immigration(t-1)
  // 3. Coral Growth Temperature Dependence (Gaussian thermal tolerance curve):
  //    temp_effect = exp(-((sst - T_opt) / T_tol)^2)
  // 4. Coral Population Dynamics (Logistic growth with interspecific competition):
  //    growth = r * temp_effect * coral_cover * (1 - (fast_cover + slow_cover) / K)
  //    coral(t) = coral(t-1) + growth - predation_loss - natural_mortality_loss
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  for (int t = 1; t < n_steps; ++t) {
    // Use states from the previous time step for calculations, ensuring they are non-negative
    Type prev_cots = CppAD::CondExpGe(cots_pred(t-1), Type(0.0), cots_pred(t-1), Type(0.0));
    Type prev_fast = CppAD::CondExpGe(fast_pred(t-1), Type(0.0), fast_pred(t-1), Type(0.0));
    Type prev_slow = CppAD::CondExpGe(slow_pred(t-1), Type(0.0), slow_pred(t-1), Type(0.0));

    // COTS Dynamics
    Type pred_on_fast = pred_rate_fast * prev_fast / (h_fast + prev_fast + Type(1e-8));
    Type pred_on_slow = pred_rate_slow * prev_slow / (h_slow + prev_slow + Type(1e-8));
    Type cots_growth = prev_cots * assim_eff * (pred_on_fast + pred_on_slow);
    Type cots_mortality = prev_cots * m_cots;
    cots_pred(t) = prev_cots + cots_growth - cots_mortality + cotsimm_dat(t-1);

    // Fast Coral Dynamics
    Type temp_effect_fast = exp(-pow((sst_dat(t-1) - T_opt_fast) / T_tol_fast, 2));
    Type fast_growth = r_fast * temp_effect_fast * prev_fast * (1 - (prev_fast + prev_slow) / (K + Type(1e-8)));
    Type fast_predation_loss = prev_cots * pred_on_fast;
    Type fast_mortality = prev_fast * m_fast;
    fast_pred(t) = prev_fast + fast_growth - fast_predation_loss - fast_mortality;

    // Slow Coral Dynamics
    Type temp_effect_slow = exp(-pow((sst_dat(t-1) - T_opt_slow) / T_tol_slow, 2));
    Type slow_growth = r_slow * temp_effect_slow * prev_slow * (1 - (prev_fast + prev_slow) / (K + Type(1e-8)));
    Type slow_predation_loss = prev_cots * pred_on_slow;
    Type slow_mortality = prev_slow * m_slow;
    slow_pred(t) = prev_slow + slow_growth - slow_predation_loss - slow_mortality;

    // Ensure current predictions are non-negative to maintain biological realism
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // LIKELIHOOD
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  // Use a lognormal distribution for strictly positive abundance/cover data.
  // Add a small constant to predictions to avoid log(0) when calculating likelihood.
  for (int t = 0; t < n_steps; ++t) {
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t) + Type(1e-8)), sd_cots, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t) + Type(1e-8)), sd_fast, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t) + Type(1e-8)), sd_slow, true);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // REPORTING
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  ADREPORT(sd_cots);
  ADREPORT(sd_fast);
  ADREPORT(sd_slow);

  return nll;
}
