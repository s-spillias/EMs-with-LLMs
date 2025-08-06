#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);           // Time vector
  DATA_VECTOR(cots_dat);       // COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);       // Fast coral cover (%)
  DATA_VECTOR(slow_dat);       // Slow coral cover (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);    // COTS larval immigration (individuals/m2/year)

  // Parameters
  PARAMETER(r_cots);           // Maximum COTS growth rate
  PARAMETER(temp_opt);         // Optimal temperature for reproduction
  PARAMETER(temp_width);       // Temperature tolerance width
  PARAMETER(K_cots);           // COTS carrying capacity
  PARAMETER(h_fast);           // Half-saturation for fast coral
  PARAMETER(h_slow);           // Half-saturation for slow coral
  PARAMETER(g_fast);           // Fast coral growth rate
  PARAMETER(g_slow);           // Slow coral growth rate
  PARAMETER(alpha_fast);       // Preference for fast coral
  PARAMETER(alpha_slow);       // Preference for slow coral
  PARAMETER(A_cots);          // Allee threshold for COTS
  PARAMETER(sigma_cots);       // COTS observation error
  PARAMETER(sigma_fast);       // Fast coral observation error
  PARAMETER(sigma_slow);       // Slow coral observation error

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize predicted vectors
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Small constant to prevent division by zero
  Type eps = Type(1e-8);

  // Process model
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on reproduction (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_width, 2));
    
    // 2. Resource availability (total coral)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. Weighted functional response for coral consumption
    Type f_fast = (alpha_fast * fast_pred(t-1)) / (h_fast + fast_pred(t-1));
    Type f_slow = (alpha_slow * slow_pred(t-1)) / (h_slow + slow_pred(t-1));
    Type total_feeding = f_fast + f_slow;
    
    // 4. COTS population dynamics with Allee effect
    Type N = cots_pred(t-1);
    Type allee_effect = CppAD::CondExpGt(N, Type(0.0),
                                        Type(1.0) - exp(-N/A_cots),
                                        Type(0.0));
    Type growth = r_cots * temp_effect * total_feeding * allee_effect;
    Type mortality = (N / K_cots) * total_feeding;
    cots_pred(t) = N + growth * N - mortality * N + cotsimm_dat(t-1);
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0), cots_pred(t), Type(0));

    // 5. Coral dynamics
    // Fast-growing coral
    Type fast_consumed = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / (h_fast + fast_pred(t-1));
    fast_pred(t) = fast_pred(t-1) + g_fast * fast_pred(t-1) * (Type(100) - fast_pred(t-1)) / Type(100) - fast_consumed;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0), fast_pred(t), Type(0));
    
    // Slow-growing coral
    Type slow_consumed = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / (h_slow + slow_pred(t-1));
    slow_pred(t) = slow_pred(t-1) + g_slow * slow_pred(t-1) * (Type(100) - slow_pred(t-1)) / Type(100) - slow_consumed;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0), slow_pred(t), Type(0));
  }

  // Observation model (log-normal)
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent log(0)
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
