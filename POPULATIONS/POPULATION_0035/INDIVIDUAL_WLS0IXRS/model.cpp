#include <TMB.hpp>

template<class Type>
Type dlnorm_custom(Type x, Type meanlog, Type sd, int give_log) {
  Type log_density = -log(x) - log(sd) - Type(0.5)*log(2.0 * M_PI) - Type(0.5)*pow((log(x) - meanlog) / sd, Type(2.0));
  return give_log ? log_density : exp(log_density);
}

// 1. Data and parameters input:
//    - DATA_VECTOR: time series data (years)
//    - DATA_VECTOR: cots_dat: observed COTS densities (individuals/m2)
//    - DATA_VECTOR: fast_dat: observed fast-growing coral cover (%)
//    - DATA_VECTOR: slow_dat: observed slow-growing coral cover (%)
//    - DATA_VECTOR: sst_dat: sea-surface temperature (°C)
//    - DATA_VECTOR: cotsimm_dat: larval immigration rate (individuals/m2/year)
//    - DATA_SCALAR: min_sd: minimum standard deviation to ensure numerical stability

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(time);               // Time (Year)
  DATA_VECTOR(cots_dat);           // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);           // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);           // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);            // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);        // Larval immigration rate (individuals/m2/year)
  DATA_SCALAR(min_sd);             // Minimum standard deviation for numerical stability

  int n = time.size();             // Number of time steps
  
  // 2. Parameters (log-transformed for positivity)
  //    Equation (description):
  //    1. COTS Dynamics:   cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1]*(1 - cots_pred[t-1] / resource_lim) - (pred_eff_fast*fast_prev + pred_eff_slow*slow_prev)*cots_pred[t-1]
  //    2. Fast Coral:      fast_pred[t] = fast_pred[t-1] - pred_eff_fast * cots_pred[t-1] * fast_pred[t-1]
  //    3. Slow Coral:      slow_pred[t] = slow_pred[t-1] - pred_eff_slow * cots_pred[t-1] * slow_pred[t-1]
  PARAMETER(log_growth_rate);      // Log intrinsic growth rate for COTS (log(year^-1))
  PARAMETER(log_resource_lim);     // Log saturation level or resource limitation coefficient for COTS density (log(units))
  PARAMETER(log_pred_eff_fast);    // Log predation efficiency on fast-growing corals (log(rate))
  PARAMETER(log_pred_eff_slow);    // Log predation efficiency on slow-growing corals (log(rate))

  // Transform to natural scale
  Type growth_rate = exp(log_growth_rate);        // COTS intrinsic growth rate (year^-1)
  Type resource_lim = exp(log_resource_lim);      // Resource limitation threshold (units)
  Type pred_eff_fast = exp(log_pred_eff_fast);      // Predation efficiency on fast corals
  Type pred_eff_slow = exp(log_pred_eff_slow);      // Predation efficiency on slow corals

  // 3. Model predictions initialization (using _pred suffix for report)
  vector<Type> cots_pred(n);     // Predicted COTS abundance
  vector<Type> fast_pred(n);     // Predicted fast-growing coral cover
  vector<Type> slow_pred(n);     // Predicted slow-growing coral cover
  
  // Initialize predictions with first observed values
  cots_pred(0)   = cots_dat(0);   // Initial adult COTS density
  fast_pred(0)   = fast_dat(0);   // Initial fast coral cover
  slow_pred(0)   = slow_dat(0);   // Initial slow coral cover

  Type nll = 0;  // Negative log likelihood

  // 4. Model Dynamics:
  // (1) COTS dynamics: combines intrinsic growth (with saturation) and losses due to predation on corals.
  // (2) Fast coral dynamics: reduction due to COTS predation.
  // (3) Slow coral dynamics: similar reduction affected by COTS predation.
  // Smooth transitions and small constants (1e-8) are used to avoid division by zero.
  for(int t = 1; t < n; t++){
    Type cots_prev = cots_pred(t-1);  // Use previous time step to avoid data leakage
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    
    // Equation 1: COTS dynamics
    cots_pred(t) = cots_prev 
                   + growth_rate * cots_prev * (1 - cots_prev / (resource_lim + Type(1e-8))) 
                   - (pred_eff_fast * fast_prev + pred_eff_slow * slow_prev) * cots_prev;
    
    // Equation 2: Fast coral dynamics
    fast_pred(t) = fast_prev - pred_eff_fast * cots_prev * fast_prev;
    
    // Equation 3: Slow coral dynamics
    slow_pred(t) = slow_prev - pred_eff_slow * cots_prev * slow_prev;
    
    // Ensure predictions remain non-negative
    cots_pred(t) = cots_pred(t) + Type(1e-8);
    fast_pred(t) = fast_pred(t) + Type(1e-8);
    slow_pred(t) = slow_pred(t) + Type(1e-8);
  }
  
  // 5. Likelihood Calculation:
  // Log-likelihood using lognormal distributions:
  // - Data are strictly positive so log-transformation is used.
  // - Minimum SD (min_sd) prevents numerical issues when variances are small.
  for(int t = 0; t < n; t++){
    Type sd_cots = (min_sd > Type(0.1) ? min_sd : Type(0.1));
    Type sd_fast = (min_sd > Type(0.1) ? min_sd : Type(0.1));
    Type sd_slow = (min_sd > Type(0.1) ? min_sd : Type(0.1));
    
    nll -= dlnorm_custom(cots_dat(t), log(cots_pred(t)), sd_cots, true);  // COTS likelihood
    nll -= dlnorm_custom(fast_dat(t), log(fast_pred(t)), sd_fast, true);   // Fast coral likelihood
    nll -= dlnorm_custom(slow_dat(t), log(slow_pred(t)), sd_slow, true);     // Slow coral likelihood
  }
  
  // 6. Reporting model predictions for diagnostics
  REPORT(cots_pred);   // Report predicted COTS dynamics
  REPORT(fast_pred);   // Report predicted fast-growing coral cover
  REPORT(slow_pred);   // Report predicted slow-growing coral cover
  
  return nll;
}
