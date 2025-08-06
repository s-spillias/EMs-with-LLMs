#include <TMB.hpp>

//------------------------------------------------------------------------------
// Template Model Builder (TMB) model for predicting Crown-of-Thorns starfish outbreaks 
// and their impact on coral communities.
// 
// Equation 1: Starfish dynamics as logistic growth with external immigration:
//   cots[t] = cots[t-1] + r_cots * cots[t-1]*(1 - cots[t-1] / K_cots) + cotsimm_dat[t-1]
// Equation 2: Slow-growing coral dynamics with logistic growth and saturating predation loss:
//   slow[t] = slow[t-1] + r_slow * slow[t-1]*(1 - slow[t-1] / K_slow) 
//             - (eff_pred_slow * cots[t-1] * slow[t-1] / (slow[t-1] + 1e-8)) * (1 + beta_sst * sst_dat[t-1])
// Equation 3: Fast-growing coral dynamics with logistic growth and saturating predation loss:
//   fast[t] = fast[t-1] + r_fast * fast[t-1]*(1 - fast[t-1] / K_fast) 
//             - (eff_pred_fast * cots[t-1] * fast[t-1] / (fast[t-1] + 1e-8)) * (1 + beta_sst * sst_dat[t-1])
// 
// All predictions (_pred) are generated using the state from the previous time step only.
// Observation likelihood is computed using lognormal errors, with a fixed minimum SD 
// to avoid numerical issues.
//------------------------------------------------------------------------------

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data declarations (time series and forcing data)
  DATA_VECTOR(Year);                        // Time vector (Year)
  DATA_VECTOR(cots_dat);                    // Observed COTS (individuals/m2)
  DATA_VECTOR(slow_dat);                    // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);                    // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);                     // Sea Surface Temperature (°C) from forcing data
  DATA_VECTOR(cotsimm_dat);                 // Crown-of-Thorns immigration rate (individuals/m2/year)
  
  // Parameter declarations with comments (units provided)
  PARAMETER(r_cots);        // Intrinsic growth rate of starfish (year^-1)
  PARAMETER(K_cots);        // Carrying capacity of starfish (individuals/m2)
  PARAMETER(eff_pred_slow); // Predation efficiency on slow-growing coral (unitless rate)
  PARAMETER(eff_pred_fast); // Predation efficiency on fast-growing coral (unitless rate)
  PARAMETER(r_slow);        // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_slow);        // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(r_fast);        // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(K_fast);        // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(beta_sst);      // Sea surface temperature effect coefficient on predation (°C^-1)
  PARAMETER(alpha_comp);    // Competition coefficient between slow and fast corals (dimensionless)
  
  // Log-scale standard deviations for lognormal likelihood (fixed minimum SD)
  PARAMETER(log_sigma_cots); // Log standard deviation for COTS observations
  PARAMETER(log_sigma_slow); // Log standard deviation for slow coral observations
  PARAMETER(log_sigma_fast); // Log standard deviation for fast coral observations
  
  // Transform log standard deviations to standard deviations
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  
  // Small constant to ensure numerical stability (avoid division by zero)
  Type epsilon = Type(1e-8);
  
  
  // Number of time steps
  int n = Year.size();
  
  // Initialize state prediction vectors
  vector<Type> cots_pred(n); // Predicted starfish population
  vector<Type> slow_pred(n); // Predicted slow-growing coral cover
  vector<Type> fast_pred(n); // Predicted fast-growing coral cover
  
  // Set initial conditions from first observation (assumed known; no likelihood penalty)
  cots_pred[0] = cots_dat[0]; // Initial COTS value
  slow_pred[0] = slow_dat[0]; // Initial slow coral cover
  fast_pred[0] = fast_dat[0]; // Initial fast coral cover
  
  // Initialize negative log likelihood
  Type nll = 0.0;
  
  // Dynamic model loop: calculate predictions using previous time step values
  for (int t = 1; t < n; t++) {
    // Equation 1: Starfish (COTS) population dynamics
    cots_pred[t] = cots_pred[t-1] 
                   + r_cots * cots_pred[t-1] * (Type(1.0) - cots_pred[t-1] / (K_cots + epsilon)) // Logistic growth term
                   + cotsimm_dat[t-1]; // Immigration forcing

    // Equation 2: Slow-growing coral dynamics with saturating predation loss and interspecific competition
    slow_pred[t] = slow_pred[t-1]
                   + r_slow * slow_pred[t-1] * (Type(1.0) - (slow_pred[t-1] + alpha_comp * fast_pred[t-1]) / (K_slow + epsilon)) // Logistic growth with competition
                   - (eff_pred_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + epsilon)) * (Type(1.0) + beta_sst * sst_dat[t-1]); // Predation loss
    
    // Equation 3: Fast-growing coral dynamics with saturating predation loss and interspecific competition
    fast_pred[t] = fast_pred[t-1]
                   + r_fast * fast_pred[t-1] * (Type(1.0) - (fast_pred[t-1] + alpha_comp * slow_pred[t-1]) / (K_fast + epsilon)) // Logistic growth with competition
                   - (eff_pred_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + epsilon)) * (Type(1.0) + beta_sst * sst_dat[t-1]); // Predation loss

    // Ensure predictions remain positive using std::max
    cots_pred[t] = std::max(cots_pred[t], epsilon);
    slow_pred[t] = std::max(slow_pred[t], epsilon);
    fast_pred[t] = std::max(fast_pred[t], epsilon);

    // Likelihood calculation: using lognormal error for each observed variable at time t
    nll -= dnorm(log(cots_dat[t] + epsilon), log(cots_pred[t] + epsilon), sigma_cots, true);
    nll -= dnorm(log(slow_dat[t] + epsilon), log(slow_pred[t] + epsilon), sigma_slow, true);
    nll -= dnorm(log(fast_dat[t] + epsilon), log(fast_pred[t] + epsilon), sigma_fast, true);
  }
  
  // Report predicted time series to enable post-analysis
  REPORT(cots_pred);  // Report starfish predictions
  REPORT(slow_pred);  // Report slow-growing coral predictions
  REPORT(fast_pred);  // Report fast-growing coral predictions
  
  return nll;
}
