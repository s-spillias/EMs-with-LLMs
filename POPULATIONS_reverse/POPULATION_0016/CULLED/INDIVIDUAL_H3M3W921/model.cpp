/*
Equations description:
1. COTS dynamics:
   cots_pred(t) = cots_pred(t-1) + [r_COTS * cots_pred(t-1) * ( (slow_pred(t-1)+fast_pred(t-1)) / (half_sat + slow_pred(t-1)+fast_pred(t-1) + 1e-8) ) * env - m_COTS * cots_pred(t-1)]
   - r_COTS: reproduction rate (year^-1)
   - m_COTS: mortality rate (year^-1)
   - env: environmental modifier (unitless)
2. Slow coral dynamics:
   slow_pred(t) = slow_pred(t-1) + growth_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow)
                   - (cots_pred(t-1)*slow_pred(t-1))/(half_sat + slow_pred(t-1) + 1e-8)
   - growth_slow: intrinsic growth rate (year^-1)
   - K_slow: carrying capacity (units corresponding to coral cover)
3. Fast coral dynamics:
   fast_pred(t) = fast_pred(t-1) + growth_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast)
                   - (cots_pred(t-1)*fast_pred(t-1))/(half_sat + fast_pred(t-1) + 1e-8)
   - growth_fast: intrinsic growth rate (year^-1)
   - K_fast: carrying capacity
Numerical constants (1e-8) are added to avoid division by zero.
Only past time-step values are used in predictions to prevent data leakage.
*/

#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs: each DATA_VECTOR should match the column names from the observations data file.
  DATA_VECTOR(Year);       // Year [integer]
  DATA_VECTOR(cots_dat);     // Observed COTS density (indiv/m^2)
  DATA_VECTOR(slow_dat);     // Observed slow coral cover (Faviidae/Porites, in %)
  DATA_VECTOR(fast_dat);     // Observed fast coral cover (Acropora spp., in %)
  DATA_VECTOR(sst_dat);      // Sea-surface temperature in Celsius
  DATA_VECTOR(cotsimm_dat);  // COTS larval immigration rate (indiv/m^2/year)

  int n = Year.size();
  
  // Model parameters (all using log-transformed values for stability)
  PARAMETER(log_r_COTS);      // Log reproduction rate for COTS (year^-1), from literature or estimation
  PARAMETER(log_m_COTS);      // Log mortality rate for COTS (year^-1)
  PARAMETER(log_growth_slow); // Log intrinsic growth rate for slow coral (year^-1)
  PARAMETER(log_growth_fast); // Log intrinsic growth rate for fast coral (year^-1)
  PARAMETER(log_K_slow);      // Log carrying capacity for slow coral (coral cover units)
  PARAMETER(log_K_fast);      // Log carrying capacity for fast coral (coral cover units)
  PARAMETER(log_half_sat);    // Log half-saturation constant for coral predation effect (matching coral cover units)
  PARAMETER(log_env);         // Log environmental modifier for COTS reproduction (unitless)

  // Observation error parameters (log-transformed to ensure positivity)
  PARAMETER(log_sd_COTS);     // Log standard deviation for COTS observations
  PARAMETER(log_sd_slow);     // Log standard deviation for slow coral observations
  PARAMETER(log_sd_fast);     // Log standard deviation for fast coral observations

  // Transform parameters from log scale
  Type r_COTS    = exp(log_r_COTS);
  Type m_COTS    = exp(log_m_COTS);
  Type growth_slow = exp(log_growth_slow);
  Type growth_fast = exp(log_growth_fast);
  Type K_slow    = exp(log_K_slow);
  Type K_fast    = exp(log_K_fast);
  Type half_sat  = exp(log_half_sat);
  Type env       = exp(log_env);
  Type sd_COTS   = exp(log_sd_COTS) + Type(1e-8);
  Type sd_slow   = exp(log_sd_slow) + Type(1e-8);
  Type sd_fast   = exp(log_sd_fast) + Type(1e-8);

  // Vectors to store model predictions
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);

  // Initialize predictions with the first observation (acting as the initial condition)
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);

  Type nll = 0.0;
  // Loop through time steps (starting from t=1; we only use previous time-step values)
  for(int t = 1; t < n; t++){
    // Equation 1: COTS dynamics
    Type coral_total = slow_pred(t-1) + fast_pred(t-1);
    // Coral modifier: saturating effect of available coral cover on reproduction
    Type coral_modifier = coral_total / (half_sat + coral_total + Type(1e-8));
    // Reproduction term and mortality term (both scaled by previous COTS density)
    Type reproduction = r_COTS * cots_pred(t-1) * coral_modifier * env;
    cots_pred(t) = cots_pred(t-1) + ( reproduction - m_COTS * cots_pred(t-1) );

    // Equation 2: Slow coral dynamics with logistic growth and COTS predation
    Type predation_slow = (cots_pred(t-1) * slow_pred(t-1)) / (half_sat + slow_pred(t-1) + Type(1e-8));
    slow_pred(t) = slow_pred(t-1) + growth_slow * slow_pred(t-1) * (1 - slow_pred(t-1) / K_slow) - predation_slow;

    // Equation 3: Fast coral dynamics with logistic growth and COTS predation
    Type predation_fast = (cots_pred(t-1) * fast_pred(t-1)) / (half_sat + fast_pred(t-1) + Type(1e-8));
    fast_pred(t) = fast_pred(t-1) + growth_fast * fast_pred(t-1) * (1 - fast_pred(t-1) / K_fast) - predation_fast;

    // Likelihood: assuming observations come from a normal distribution around model predictions
    nll -= dnorm(cots_dat(t), cots_pred(t), sd_COTS, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
  }

  // REPORT predictions so that they can be output and inspected
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
