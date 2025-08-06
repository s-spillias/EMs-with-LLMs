#include <TMB.hpp> // TMB header for automatic differentiation

// TMB model for Crown of Thorns starfish dynamics with coral interactions
TMB_OBJECTIVE_FUNCTION(){
  
  // DATA INPUTS:
  DATA_VECTOR(Year);       // Time vector (years)
  DATA_VECTOR(cots_dat);   // Observed Crown-Of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(slow_dat);   // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);   // Observed fast-growing coral cover (%)
  
  // PARAMETERS:
  PARAMETER(log_growth_cots);        // Log intrinsic growth rate of starfish (log(year^-1))
  PARAMETER(log_carrying_capacity);    // Log carrying capacity of starfish (log(individuals/m2))
  PARAMETER(log_pred_rate_slow);       // Log predation rate impacting slow-growing corals (log(rate))
  PARAMETER(log_pred_rate_fast);       // Log predation rate impacting fast-growing corals (log(rate))
  PARAMETER(efficiency);               // Efficiency of predation (unitless, between 0 and 1)
  PARAMETER(beta_temp);                // Coefficient for environmental (SST) effect on starfish growth
  
  PARAMETER(log_sigma_cots);           // Log standard deviation for starfish observation error
  PARAMETER(log_sigma_slow);           // Log standard deviation for slow coral observation error
  PARAMETER(log_sigma_fast);           // Log standard deviation for fast coral observation error
  
  // TRANSFORM PARAMETERS:
  Type growth_cots = exp(log_growth_cots);      
  Type carrying_capacity = exp(log_carrying_capacity); 
  Type pred_rate_slow = exp(log_pred_rate_slow);  
  Type pred_rate_fast = exp(log_pred_rate_fast);  
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8);  
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8);
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8);
  
  int n = Year.size();
  
  // STATE VECTORS FOR PREDICTIONS:
  vector<Type> cots_pred(n); 
  vector<Type> slow_pred(n); 
  vector<Type> fast_pred(n); 
  
  // INITIALIZATION (using previous data to seed predictions)
  cots_pred[0] = cots_dat[0];  
  slow_pred[0] = slow_dat[0];  
  fast_pred[0] = fast_dat[0];  
  
  // NEGATIVE LOG LIKELIHOOD:
  Type nll = 0;
  
  for (int t = 1; t < n; t++){
    // Equation 1: Starfish dynamics
    //   Logistic growth modulated by environmental forcing and predation on corals.
    Type temp_effect = exp(beta_temp * Type(0)); // Placeholder for environmental forcing
    cots_pred[t] = cots_pred[t-1] 
                 + growth_cots * cots_pred[t-1] * (Type(1) - cots_pred[t-1] / (carrying_capacity + Type(1e-8))) * temp_effect
                 - efficiency * (pred_rate_slow * slow_pred[t-1] + pred_rate_fast * fast_pred[t-1]);
    
    // Equation 2: Slow coral dynamics (Faviidae/Porites)
    //   Logistic regrowth with reduction due to starfish predation.
    Type growth_rate_slow = Type(0.05); // Assumed regrowth rate for slow coral (% per year)
    slow_pred[t] = slow_pred[t-1] 
                 + growth_rate_slow * slow_pred[t-1] * (Type(1) - slow_pred[t-1] / Type(100))
                 - pred_rate_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8));
    
    // Equation 3: Fast coral dynamics (Acropora)
    //   Faster logistic regrowth with reduction due to starfish predation.
    Type growth_rate_fast = Type(0.1); // Assumed regrowth rate for fast coral (% per year)
    fast_pred[t] = fast_pred[t-1] 
                 + growth_rate_fast * fast_pred[t-1] * (Type(1) - fast_pred[t-1] / Type(100))
                 - pred_rate_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8));
    
    // LIKELIHOOD CALCULATION:
    // Lognormal observation error for strictly positive data.
    nll -= dnorm(log(cots_dat[t] + Type(1e-8)), log(cots_pred[t] + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), sigma_slow, true);
    nll -= dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), sigma_fast, true);
  }
  
  // REPORT predictions for model diagnostics:
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
#include <TMB.hpp> // TMB header for automatic differentiation

// TMB model for Crown of Thorns starfish dynamics with coral interactions
TMB_OBJECTIVE_FUNCTION(){
  
  // DATA INPUTS:
  DATA_VECTOR(Year);       // Time vector (years)
  DATA_VECTOR(cots_dat);   // Observed Crown-Of-Thorns starfish abundance (individuals/m2)
  DATA_VECTOR(slow_dat);   // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);   // Observed fast-growing coral cover (%)
  
  // PARAMETERS:
  PARAMETER(log_growth_cots);        // Log intrinsic growth rate of starfish (log(year^-1))
  PARAMETER(log_carrying_capacity);    // Log carrying capacity of starfish (log(individuals/m2))
  PARAMETER(log_pred_rate_slow);       // Log predation rate impacting slow-growing corals (log(rate))
  PARAMETER(log_pred_rate_fast);       // Log predation rate impacting fast-growing corals (log(rate))
  PARAMETER(efficiency);               // Efficiency of predation (unitless, between 0 and 1)
  PARAMETER(beta_temp);                // Coefficient for environmental (SST) effect on starfish growth
  
  PARAMETER(log_sigma_cots);           // Log standard deviation for starfish observation error
  PARAMETER(log_sigma_slow);           // Log standard deviation for slow coral observation error
  PARAMETER(log_sigma_fast);           // Log standard deviation for fast coral observation error
  
  // TRANSFORM PARAMETERS:
  Type growth_cots = exp(log_growth_cots);      
  Type carrying_capacity = exp(log_carrying_capacity); 
  Type pred_rate_slow = exp(log_pred_rate_slow);  
  Type pred_rate_fast = exp(log_pred_rate_fast);  
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8);  
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8);
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8);
  
  int n = Year.size();
  
  // STATE VECTORS FOR PREDICTIONS:
  vector<Type> cots_pred(n); 
  vector<Type> slow_pred(n); 
  vector<Type> fast_pred(n); 
  
  // INITIALIZATION (using previous data to seed predictions)
  cots_pred[0] = cots_dat[0];  
  slow_pred[0] = slow_dat[0];  
  fast_pred[0] = fast_dat[0];  
  
  // NEGATIVE LOG LIKELIHOOD:
  Type nll = 0;
  
  for (int t = 1; t < n; t++){
    // Equation 1: Starfish dynamics
    //   Logistic growth modulated by environmental forcing and predation on corals.
    Type temp_effect = exp(beta_temp * Type(0)); // Placeholder for environmental forcing
    cots_pred[t] = cots_pred[t-1] 
                 + growth_cots * cots_pred[t-1] * (Type(1) - cots_pred[t-1] / (carrying_capacity + Type(1e-8))) * temp_effect
                 - efficiency * (pred_rate_slow * slow_pred[t-1] + pred_rate_fast * fast_pred[t-1]);
    
    // Equation 2: Slow coral dynamics (Faviidae/Porites)
    //   Logistic regrowth with reduction due to starfish predation.
    Type growth_rate_slow = Type(0.05); // Assumed regrowth rate for slow coral (% per year)
    slow_pred[t] = slow_pred[t-1] 
                 + growth_rate_slow * slow_pred[t-1] * (Type(1) - slow_pred[t-1] / Type(100))
                 - pred_rate_slow * cots_pred[t-1] * slow_pred[t-1] / (slow_pred[t-1] + Type(1e-8));
    
    // Equation 3: Fast coral dynamics (Acropora)
    //   Faster logistic regrowth with reduction due to starfish predation.
    Type growth_rate_fast = Type(0.1); // Assumed regrowth rate for fast coral (% per year)
    fast_pred[t] = fast_pred[t-1] 
                 + growth_rate_fast * fast_pred[t-1] * (Type(1) - fast_pred[t-1] / Type(100))
                 - pred_rate_fast * cots_pred[t-1] * fast_pred[t-1] / (fast_pred[t-1] + Type(1e-8));
    
    // LIKELIHOOD CALCULATION:
    // Lognormal observation error for strictly positive data.
    nll -= dnorm(log(cots_dat[t] + Type(1e-8)), log(cots_pred[t] + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), sigma_slow, true);
    nll -= dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), sigma_fast, true);
  }
  
  // REPORT predictions for model diagnostics:
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
