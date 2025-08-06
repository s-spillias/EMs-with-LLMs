#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Helper functions
  auto bound = [](Type x, Type lower, Type upper) {
    Type z = 2.0 * (x - lower)/(upper - lower) - 1.0;
    return (lower + upper)/2.0 + (upper - lower)/2.0 * z/sqrt(1.0 + z*z);
  };
  // Data
  DATA_VECTOR(Year);             // Time vector (years)
  DATA_VECTOR(cots_dat);         // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);         // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);         // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);          // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);      // COTS larval immigration rate (individuals/m^2/year)

  // Parameters
  PARAMETER(r_cots);             // COTS intrinsic growth rate
  PARAMETER(K_cots);             // COTS carrying capacity
  PARAMETER(r_fast);             // Fast-growing coral growth rate
  PARAMETER(r_slow);             // Slow-growing coral growth rate
  PARAMETER(K_fast);             // Fast-growing coral carrying capacity
  PARAMETER(K_slow);             // Slow-growing coral carrying capacity
  PARAMETER(a_fast);             // Attack rate on fast coral
  PARAMETER(a_slow);             // Attack rate on slow coral
  PARAMETER(h_fast);             // Handling time for fast coral
  PARAMETER(h_slow);             // Handling time for slow coral
  PARAMETER(T_opt);              // Optimal temperature for COTS
  PARAMETER(T_width);            // Temperature tolerance width
  PARAMETER(A_thresh);           // Allee effect threshold
  PARAMETER(sigma_cots);         // Observation error SD for COTS
  PARAMETER(sigma_fast);         // Observation error SD for fast coral
  PARAMETER(sigma_slow);         // Observation error SD for slow coral

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  const Type max_val = Type(1e3);

  // Transform parameters to ensure valid bounds
  Type r_cots_bounded = bound(r_cots, Type(0.01), Type(2.0));
  Type r_fast_bounded = bound(r_fast, Type(0.01), Type(1.0));
  Type r_slow_bounded = bound(r_slow, Type(0.01), Type(0.5));
  Type K_cots_bounded = bound(K_cots, Type(0.1), Type(5.0));
  Type K_fast_bounded = bound(K_fast, Type(1.0), Type(100.0));
  Type K_slow_bounded = bound(K_slow, Type(1.0), Type(50.0));
  Type a_fast_bounded = bound(a_fast, Type(0.01), Type(1.0));
  Type a_slow_bounded = bound(a_slow, Type(0.01), Type(1.0));
  Type h_fast_bounded = bound(h_fast, Type(0.01), Type(1.0));
  Type h_slow_bounded = bound(h_slow, Type(0.01), Type(1.0));
  Type T_width_bounded = bound(T_width, Type(0.1), Type(5.0));
  Type A_thresh_bounded = bound(A_thresh, Type(0.01), Type(1.0));
  Type sigma_cots_bounded = bound(sigma_cots, Type(0.01), Type(1.0));
  Type sigma_fast_bounded = bound(sigma_fast, Type(0.01), Type(1.0));
  Type sigma_slow_bounded = bound(sigma_slow, Type(0.01), Type(1.0));

  // Vectors to store predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> slow_pred(Year.size());

  // Initialize first predictions with observations
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Time series simulation
  for(int t = 1; t < Year.size(); t++) {
    // Temperature effect on reproduction (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - T_opt, 2.0)/(2.0 * pow(T_width_bounded, 2.0)));
    
    // Allee effect combined with carrying capacity
    Type allee_effect = cots_pred(t-1)/(cots_pred(t-1) + A_thresh_bounded);
    Type density_effect = (Type(1.0) - cots_pred(t-1)/K_cots_bounded);
    
    // Modified COTS growth with Allee effect and temperature dependence
    Type cots_growth = r_cots_bounded * cots_pred(t-1) * allee_effect * density_effect * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = bound(cots_pred(t), Type(eps), K_cots_bounded);
    
    // Linear functional response for simplicity
    Type F_fast = a_fast_bounded * cots_pred(t-1);
    Type F_slow = a_slow_bounded * cots_pred(t-1);
    
    // Coral dynamics with logistic growth
    Type fast_growth = r_fast_bounded * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast_bounded);
    Type slow_growth = r_slow_bounded * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow_bounded);
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - F_fast;
    slow_pred(t) = slow_pred(t-1) + slow_growth - F_slow;
    
    // Bound coral predictions
    fast_pred(t) = bound(fast_pred(t), Type(eps), K_fast_bounded);
    slow_pred(t) = bound(slow_pred(t), Type(eps), K_slow_bounded);
  }

  // Observation model using log-normal distribution with bounded SDs
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to avoid taking log of zero
    Type obs_cots = max(cots_dat(t), eps);
    Type pred_cots = max(cots_pred(t), eps);
    Type obs_fast = max(fast_dat(t), eps);
    Type pred_fast = max(fast_pred(t), eps);
    Type obs_slow = max(slow_dat(t), eps);
    Type pred_slow = max(slow_pred(t), eps);
    
    nll -= dnorm(log(obs_cots), log(pred_cots), sigma_cots_bounded, true);
    nll -= dnorm(log(obs_fast), log(pred_fast), sigma_fast_bounded, true);
    nll -= dnorm(log(obs_slow), log(pred_slow), sigma_slow_bounded, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
