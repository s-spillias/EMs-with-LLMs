#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);     // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);      // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);  // COTS immigration rate (individuals/m²/year)
  
  // Parameters
  PARAMETER(r_slow);         // Growth rate of slow-growing corals
  PARAMETER(r_fast);         // Growth rate of fast-growing corals
  PARAMETER(K_slow);         // Carrying capacity of slow-growing corals
  PARAMETER(K_fast);         // Carrying capacity of fast-growing corals
  PARAMETER(alpha_slow);     // COTS predation rate on slow corals
  PARAMETER(alpha_fast);     // COTS predation rate on fast corals
  PARAMETER(beta);          // Density-dependent mortality coefficient
  PARAMETER(gamma);         // Temperature effect on recruitment
  PARAMETER(T_opt);         // Optimal temperature for recruitment
  PARAMETER(refuge_effect);  // Coral refuge effect on COTS survival
  PARAMETER(log_sigma_cots);  // Log of COTS observation error SD
  PARAMETER(log_sigma_coral); // Log of coral observation error SD
  
  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors to store predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial values
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Model equations
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature-dependent COTS recruitment and mortality
    Type temp_effect = exp(-gamma * pow(sst_dat(t) - T_opt, 2));
    temp_effect = CppAD::CondExpGe(temp_effect, Type(0.01), temp_effect, Type(0.01));
    
    // 2. COTS population dynamics with bounded predictions
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    Type refuge_recruitment = refuge_effect * total_coral * cotsimm_dat(t) / (total_coral + Type(50.0));
    refuge_recruitment = CppAD::CondExpGe(refuge_recruitment, Type(0), refuge_recruitment, Type(0));
    
    Type cots_mortality = beta * pow(cots_pred(t-1), 2) / (pow(cots_pred(t-1), 2) + Type(1.0));
    cots_mortality = CppAD::CondExpGe(cots_mortality, Type(0), cots_mortality, Type(0));
    
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * cotsimm_dat(t) +  // Base recruitment
                   refuge_recruitment -            // Additional recruitment from coral refuge
                   cots_mortality;
    
    // 3. Slow-growing coral dynamics with bounded growth
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow);
    slow_growth = CppAD::CondExpGe(slow_growth, Type(-0.5), slow_growth, Type(-0.5));
    
    Type slow_predation = alpha_slow * cots_pred(t-1) * slow_pred(t-1)/(slow_pred(t-1) + Type(1.0));
    slow_predation = CppAD::CondExpGe(slow_predation, Type(0), slow_predation, Type(0));
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    
    // 4. Fast-growing coral dynamics with bounded growth
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast);
    fast_growth = CppAD::CondExpGe(fast_growth, Type(-0.5), fast_growth, Type(-0.5));
    
    Type fast_predation = alpha_fast * cots_pred(t-1) * fast_pred(t-1)/(fast_pred(t-1) + Type(1.0));
    fast_predation = CppAD::CondExpGe(fast_predation, Type(0), fast_predation, Type(0));
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    
    // Ensure predictions stay within reasonable bounds using smooth min/max
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.01), cots_pred(t), Type(0.01));
    cots_pred(t) = CppAD::CondExpLe(cots_pred(t), Type(10.0), cots_pred(t), Type(10.0));
    
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.1), slow_pred(t), Type(0.1));
    slow_pred(t) = CppAD::CondExpLe(slow_pred(t), K_slow, slow_pred(t), K_slow);
    
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.1), fast_pred(t), Type(0.1));
    fast_pred(t) = CppAD::CondExpLe(fast_pred(t), K_fast, fast_pred(t), K_fast);
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < cots_dat.size(); t++) {
    // 5. COTS abundance likelihood
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps), 
                 sigma_cots, 
                 true);
    
    // 6. Coral cover likelihoods
    nll -= dnorm(log(slow_dat(t) + eps), 
                 log(slow_pred(t) + eps), 
                 sigma_coral, 
                 true);
    
    nll -= dnorm(log(fast_dat(t) + eps), 
                 log(fast_pred(t) + eps), 
                 sigma_coral, 
                 true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(sigma_cots);
  REPORT(sigma_coral);
  
  return nll;
}
