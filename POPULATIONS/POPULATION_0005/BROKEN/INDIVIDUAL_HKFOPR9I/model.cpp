#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_fast);                 // Fast coral carrying capacity
  PARAMETER(K_slow);                 // Slow coral carrying capacity
  PARAMETER(alpha_fast);             // COTS preference for fast coral
  PARAMETER(alpha_slow);             // COTS preference for slow coral
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_width);             // Temperature tolerance width
  PARAMETER(sigma_cots);             // COTS observation error
  PARAMETER(sigma_fast);             // Fast coral observation error
  PARAMETER(sigma_slow);             // Slow coral observation error
  PARAMETER(A_cots);                 // COTS Allee threshold

  // Initialize negative log likelihood
  Type nll = 0.0;
  
  // Initialize predicted vectors
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);

  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Model equations
  for(int i = 1; i < n; i++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(i-1) - temp_opt) / temp_width, 2));
    
    // 2. Total coral resource availability
    Type total_coral = (fast_pred(i-1) + slow_pred(i-1)) / (Type(100.0) + eps);
    
    // 3. COTS population dynamics with Allee effect
    Type N = cots_pred(i-1);
    Type allee_effect = N / (A_cots + N);  // Simpler Allee effect
    Type logistic = Type(1.0) - N/K_cots;
    
    Type cots_growth = r_cots * N * logistic * temp_effect * allee_effect;
    cots_pred(i) = N + cots_growth + cotsimm_dat(i-1);
    cots_pred(i) = CppAD::CondExpGt(cots_pred(i), eps, cots_pred(i), eps);
    
    // 4. Coral predation impact (Type II functional response)
    Type pred_impact_fast = (alpha_fast * cots_pred(i-1) * fast_pred(i-1)) / (Type(1.0) + alpha_fast * fast_pred(i-1) + alpha_slow * slow_pred(i-1));
    Type pred_impact_slow = (alpha_slow * cots_pred(i-1) * slow_pred(i-1)) / (Type(1.0) + alpha_fast * fast_pred(i-1) + alpha_slow * slow_pred(i-1));
    
    // 5. Coral dynamics
    fast_pred(i) = fast_pred(i-1) + r_fast * fast_pred(i-1) * (Type(1.0) - fast_pred(i-1)/K_fast) - pred_impact_fast;
    slow_pred(i) = slow_pred(i-1) + r_slow * slow_pred(i-1) * (Type(1.0) - slow_pred(i-1)/K_slow) - pred_impact_slow;
    
    // Bound coral cover between 0 and carrying capacity
    fast_pred(i) = CppAD::CondExpGt(fast_pred(i), Type(0.0), fast_pred(i), Type(0.0));
    fast_pred(i) = CppAD::CondExpLt(fast_pred(i), K_fast, fast_pred(i), K_fast);
    slow_pred(i) = CppAD::CondExpGt(slow_pred(i), Type(0.0), slow_pred(i), Type(0.0));
    slow_pred(i) = CppAD::CondExpLt(slow_pred(i), K_slow, slow_pred(i), K_slow);
  }
  
  // Observation model (normal on log scale)
  for(int i = 0; i < n; i++) {
    if(cots_dat(i) > eps && cots_pred(i) > eps) {
      nll -= dnorm(log(cots_dat(i)), log(cots_pred(i)), sigma_cots, true);
    }
    if(fast_dat(i) > eps && fast_pred(i) > eps) {
      nll -= dnorm(log(fast_dat(i)), log(fast_pred(i)), sigma_fast, true);
    }
    if(slow_dat(i) > eps && slow_pred(i) > eps) {
      nll -= dnorm(log(slow_dat(i)), log(slow_pred(i)), sigma_slow, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
