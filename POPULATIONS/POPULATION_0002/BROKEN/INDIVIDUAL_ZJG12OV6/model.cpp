#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Helper functions
  auto bound = [](Type x, Type lower, Type upper) {
    return lower + (upper - lower)/(1 + exp(-x));
  };
  
  // Safe max/min functions for TMB
  auto safe_max = [](Type a, Type b) {
    return (a + b + sqrt(pow(a - b, 2)))/Type(2);
  };
  
  auto safe_min = [](Type a, Type b) {
    return (a + b - sqrt(pow(a - b, 2)))/Type(2);
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
  PARAMETER(q);                  // Density-dependent predation coefficient
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
    // Temperature-dependent COTS growth with improved stability
    Type temp_diff = (sst_dat(t-1) - T_opt)/(T_width + Type(0.1));
    Type temp_effect = Type(0.1) + Type(0.9) * exp(-Type(0.5) * pow(temp_diff, 2.0));
    
    // COTS population dynamics with smoother transitions
    Type rel_density = cots_pred(t-1)/(K_cots_bounded + eps);
    Type density_factor = Type(1.0)/(Type(1.0) + exp(Type(5.0) * (rel_density - Type(1.0))));
    Type cots_growth = r_cots_bounded * temp_effect * cots_pred(t-1) * density_factor;
    
    // Update with immigration and bounds
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = eps + (K_cots_bounded - eps)/(Type(1.0) + exp(-cots_pred(t)));
    
    // Density-dependent predation with smooth saturation
    Type density_term = q * cots_pred(t-1)/(Type(1.0) + q * cots_pred(t-1));
    Type density_effect = Type(1.0) + density_term;
    
    // Predation with smooth limiting
    Type potential_F_fast = a_fast_bounded * density_effect * cots_pred(t-1);
    Type potential_F_slow = a_slow_bounded * density_effect * cots_pred(t-1);
    Type F_fast = potential_F_fast * fast_pred(t-1)/(potential_F_fast + fast_pred(t-1) + eps);
    Type F_slow = potential_F_slow * slow_pred(t-1)/(potential_F_slow + slow_pred(t-1) + eps);
    
    // Coral dynamics with smooth logistic growth
    Type fast_rel = fast_pred(t-1)/(K_fast_bounded + eps);
    Type slow_rel = slow_pred(t-1)/(K_slow_bounded + eps);
    
    // Smooth growth limitation
    Type fast_limit = Type(1.0)/(Type(1.0) + exp(Type(5.0) * (fast_rel - Type(1.0))));
    Type slow_limit = Type(1.0)/(Type(1.0) + exp(Type(5.0) * (slow_rel - Type(1.0))));
    
    Type fast_growth = r_fast_bounded * fast_pred(t-1) * fast_limit;
    Type slow_growth = r_slow_bounded * slow_pred(t-1) * slow_limit;
    
    // Update with smooth bounds
    fast_pred(t) = fast_pred(t-1) + fast_growth - F_fast;
    slow_pred(t) = slow_pred(t-1) + slow_growth - F_slow;
    
    fast_pred(t) = eps + (K_fast_bounded - eps)/(Type(1.0) + exp(-fast_pred(t)));
    slow_pred(t) = eps + (K_slow_bounded - eps)/(Type(1.0) + exp(-slow_pred(t)));
  }

  // Observation model using log-normal distribution with bounded SDs
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_bounded, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_bounded, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_bounded, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
