#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(cots_dat);             // COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);             // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);             // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration rate

  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(A_cots);                 // COTS Allee effect threshold
  PARAMETER(tau_fast);               // Fast coral recovery delay
  PARAMETER(tau_slow);               // Slow coral recovery delay
  PARAMETER(alpha_fast);             // Attack rate on fast coral
  PARAMETER(alpha_slow);             // Attack rate on slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(temp_opt);               // Optimal temperature
  PARAMETER(temp_width);             // Temperature tolerance width
  PARAMETER(sigma_cots);             // SD for COTS observations
  PARAMETER(sigma_fast);             // SD for fast coral observations
  PARAMETER(sigma_slow);             // SD for slow coral observations

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors to store predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> slow_pred(Year.size());

  // Initialize first predictions with first observations
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Time series predictions
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2) / (2 * pow(temp_width, 2)));
    
    // 2. Resource availability effect (Type II functional response)
    Type resource_effect = (fast_pred(t-1) + slow_pred(t-1)) / 
                         (fast_pred(t-1) + slow_pred(t-1) + Type(10.0));
    
    // 3. COTS population dynamics with simplified Allee effect
    Type allee_effect = cots_pred(t-1) / (A_cots + cots_pred(t-1) + eps);
    
    // Calculate carrying capacity term with smoother transition
    Type k_term = Type(1.0) / (Type(1.0) + cots_pred(t-1)/K_cots);
    
    // Calculate total growth rate with bounded multipliers
    Type growth_mult = CppAD::CondExpLe(temp_effect, Type(1.0), temp_effect, Type(1.0)) * 
                      CppAD::CondExpLe(resource_effect, Type(1.0), resource_effect, Type(1.0));
    Type total_growth = r_cots * growth_mult * allee_effect * k_term;
    
    // Update population
    cots_pred(t) = cots_pred(t-1) * (Type(1.0) + total_growth) + cotsimm_dat(t-1);
    
    // 4. Calculate delayed coral state with simpler moving average
    Type delayed_fast = fast_pred(t-1);
    Type delayed_slow = slow_pred(t-1);
    Type count_fast = Type(1.0);
    Type count_slow = Type(1.0);
    
    // Use shorter lookback to reduce complexity
    int lookback = 3;
    
    for(int i = 1; i < lookback && (t-i) > 0; i++) {
        delayed_fast += fast_pred(t-i-1);
        delayed_slow += slow_pred(t-i-1);
        count_fast += Type(1.0);
        count_slow += Type(1.0);
    }
    
    delayed_fast /= count_fast;
    delayed_slow /= count_slow;
    
    // 5. Coral predation rates (Type II functional response)
    Type fast_consumption = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / 
                          (1 + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1));
    Type slow_consumption = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / 
                          (1 + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1));
    
    // 6. Coral dynamics with delayed recovery
    Type total_cover = (delayed_fast + delayed_slow);
    Type space_limit = CppAD::CondExpGe(total_cover, Type(100.0), 
                                       Type(0.0), 
                                       Type(1.0) - total_cover/Type(100.0));
                                       
    fast_pred(t) = fast_pred(t-1) + r_fast * delayed_fast * space_limit - fast_consumption;
    slow_pred(t) = slow_pred(t-1) + r_slow * delayed_slow * space_limit - slow_consumption;
    
    // Ensure predictions stay within reasonable bounds
    cots_pred(t) = CppAD::CondExpLe(cots_pred(t), Type(10.0), 
                                   cots_pred(t), Type(10.0));
    fast_pred(t) = CppAD::CondExpLe(fast_pred(t), Type(100.0), 
                                   fast_pred(t), Type(100.0));
    slow_pred(t) = CppAD::CondExpLe(slow_pred(t), Type(100.0), 
                                   slow_pred(t), Type(100.0));
                                   
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), eps, 
                                   cots_pred(t), eps);
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), eps, 
                                   fast_pred(t), eps);
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), eps, 
                                   slow_pred(t), eps);
  }

  // Observation model using log-normal distribution
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to prevent taking log of zero
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
