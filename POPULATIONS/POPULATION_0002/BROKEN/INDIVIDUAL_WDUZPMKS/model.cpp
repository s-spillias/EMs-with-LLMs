#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
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
  PARAMETER(sigma_cots);         // Observation error SD for COTS
  PARAMETER(sigma_fast);         // Observation error SD for fast coral
  PARAMETER(sigma_slow);         // Observation error SD for slow coral

  // Transform parameters to ensure positivity
  Type r_cots_pos = exp(r_cots);
  Type K_cots_pos = exp(K_cots);
  Type r_fast_pos = exp(r_fast);
  Type r_slow_pos = exp(r_slow);
  Type K_fast_pos = exp(K_fast);
  Type K_slow_pos = exp(K_slow);
  Type a_fast_pos = exp(a_fast);
  Type a_slow_pos = exp(a_slow);
  Type sigma_cots_pos = exp(sigma_cots);
  Type sigma_fast_pos = exp(sigma_fast);
  Type sigma_slow_pos = exp(sigma_slow);

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  const Type max_val = Type(1e3);

  // Apply soft bounds through penalty terms
  Type nll_penalty = 0.0;
  nll_penalty -= dnorm(r_cots, Type(0.8), Type(0.5), true);
  nll_penalty -= dnorm(K_cots, Type(2.5), Type(1.0), true);
  nll_penalty -= dnorm(r_fast, Type(0.3), Type(0.2), true);
  nll_penalty -= dnorm(r_slow, Type(0.1), Type(0.05), true);
  nll_penalty -= dnorm(K_fast, Type(50.0), Type(20.0), true);
  nll_penalty -= dnorm(K_slow, Type(30.0), Type(10.0), true);
  nll_penalty -= dnorm(a_fast, Type(0.5), Type(0.2), true);
  nll_penalty -= dnorm(a_slow, Type(0.2), Type(0.1), true);
  nll_penalty -= dnorm(T_opt, Type(28.0), Type(2.0), true);
  
  nll += nll_penalty;

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
    // COTS growth with simplified Allee effect and temperature dependence
    Type temp_effect = exp(-pow(sst_dat(t-1) - T_opt, 2)/(2 * pow(Type(2.0), 2)));
    Type allee_term = cots_pred(t-1)/(Type(0.1) + cots_pred(t-1));
    Type density_effect = (Type(1.0) - cots_pred(t-1)/K_cots_pos);
    
    Type cots_growth = r_cots_pos * temp_effect * allee_term * cots_pred(t-1) * density_effect;
    
    // Ensure predictions stay positive and finite
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t-1) + cots_growth + cotsimm_dat(t-1),
                                   Type(eps),
                                   cots_pred(t-1) + cots_growth + cotsimm_dat(t-1),
                                   Type(eps));
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), K_cots, cots_pred(t), K_cots);
    
    // Simple linear functional response
    Type F_fast = a_fast_pos * cots_pred(t-1);
    Type F_slow = a_slow_pos * cots_pred(t-1);
    
    // Basic logistic growth for corals
    Type fast_growth = r_fast_pos * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast_pos);
    Type slow_growth = r_slow_pos * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow_pos);
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - F_fast;
    slow_pred(t) = slow_pred(t-1) + slow_growth - F_slow;
    
    // Bound coral predictions
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(eps),
                                   CppAD::CondExpLt(fast_pred(t), K_fast, fast_pred(t), K_fast),
                                   Type(eps));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(eps),
                                   CppAD::CondExpLt(slow_pred(t), K_slow, slow_pred(t), K_slow),
                                   Type(eps));
  }

  // Observation model using log-normal distribution with bounded SDs
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_pos, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_pos, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_pos, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
