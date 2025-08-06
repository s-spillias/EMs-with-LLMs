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
  PARAMETER(alpha_fast);             // Attack rate on fast coral
  PARAMETER(alpha_slow);             // Attack rate on slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(temp_opt);               // Optimal temperature
  PARAMETER(temp_width);             // Temperature tolerance width
  PARAMETER(sigma_cots);             // SD for COTS observations
  PARAMETER(sigma_fast);             // SD for fast coral observations
  PARAMETER(sigma_slow);             // SD for slow coral observations
  PARAMETER(allee_thresh);           // COTS Allee effect threshold density

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
    
    // 3. COTS population dynamics with Allee effect
    Type N = cots_pred(t-1);
    Type allee_factor = pow(N, 2.0) / (pow(N, 2.0) + pow(allee_thresh, 2.0));
    Type density_effect = Type(1.0) - N/K_cots;
    
    cots_pred(t) = N + r_cots * N * temp_effect * resource_effect * 
                   allee_factor * density_effect + cotsimm_dat(t-1);
    
    // 4. Coral predation rates (Type II functional response)
    Type fast_consumption = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / 
                          (1 + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1));
    Type slow_consumption = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / 
                          (1 + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1));
    
    // 5. Coral dynamics
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * 
                   (1 - (fast_pred(t-1) + slow_pred(t-1))/100) - fast_consumption;
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * 
                   (1 - (fast_pred(t-1) + slow_pred(t-1))/100) - slow_consumption;
    
    // 6. Bound predictions to biologically reasonable values
    cots_pred(t) = std::max(Type(0.0), std::min(cots_pred(t), Type(5.0)));
    fast_pred(t) = std::max(Type(0.0), std::min(fast_pred(t), Type(100.0)));
    slow_pred(t) = std::max(Type(0.0), std::min(slow_pred(t), Type(100.0)));
  }

  // Observation model using normal distribution
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots * cots_pred(t), true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast * fast_pred(t), true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow * slow_pred(t), true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
