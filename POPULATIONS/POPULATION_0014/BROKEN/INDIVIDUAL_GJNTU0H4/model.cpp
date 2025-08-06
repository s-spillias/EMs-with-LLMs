#include <TMB.hpp>


template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. Data declarations:
  //    Year: time steps (years) from the data file.
  //    cots_dat: observed COTS abundance (ind/m^2).
  //    fast_dat: observed fast-growing coral cover (%).
  //    slow_dat: observed slow-growing coral cover (%).
  DATA_VECTOR(Year);
  DATA_VECTOR(cots_dat);
  DATA_VECTOR(fast_dat);
  DATA_VECTOR(slow_dat);

  // 2. Parameter declarations with their biological meanings:
  //    growth_rate: intrinsic COTS growth rate (year^-1).
  //    outbreak_trigger: COTS threshold triggering outbreak conditions (ind/m^2).
  //    outbreak_efficiency: multiplier for COTS growth during outbreaks (dimensionless).
  //    coral_predation_rate_fast: predation rate on fast-growing coral (% per unit COTS).
  //    coral_predation_rate_slow: predation rate on slow-growing coral (% per unit COTS).
  //    coral_regen_fast: regeneration rate for fast-growing coral (% per year).
  //    coral_regen_slow: regeneration rate for slow-growing coral (% per year).
  //    log_sd: log standard deviation for observation errors.
  PARAMETER(growth_rate);               // year^-1 (from literature)
  PARAMETER(outbreak_trigger);          // ind/m^2 (expert opinion)
  PARAMETER(outbreak_efficiency);       // dimensionless (expert opinion)
  PARAMETER(outbreak_hill_exp);         // Hill exponent for outbreak multiplier
  PARAMETER(coral_predation_rate_fast); // per unit COTS (literature)
  PARAMETER(coral_predation_rate_slow); // per unit COTS (literature)
  PARAMETER(coral_regen_fast);          // % per year (literature)
  PARAMETER(coral_regen_slow);          // % per year (literature)
  PARAMETER(log_sd);                    // log-scale standard deviation (initial estimate)
  // Fallback default values for parameters if they are NA.
  if(std::isnan(growth_rate)) growth_rate = Type(0.5);
  if(std::isnan(outbreak_trigger)) outbreak_trigger = Type(50.0);
  if(std::isnan(outbreak_efficiency)) outbreak_efficiency = Type(2.0);
  if(std::isnan(outbreak_hill_exp)) outbreak_hill_exp = Type(2.0);
  if(std::isnan(coral_predation_rate_fast)) coral_predation_rate_fast = Type(0.01);
  if(std::isnan(coral_predation_rate_slow)) coral_predation_rate_slow = Type(0.005);
  if(std::isnan(coral_regen_fast)) coral_regen_fast = Type(0.1);
  if(std::isnan(coral_regen_slow)) coral_regen_slow = Type(0.05);
  if(std::isnan(log_sd)) log_sd = Type(-1.0);
  // 3. Initialization:
  int n = Year.size();
  vector<Type> cots_pred(n);  // predicted COTS abundance
  vector<Type> fast_pred(n);  // predicted fast-growing coral cover
  vector<Type> slow_pred(n);  // predicted slow-growing coral cover

  // Set initial conditions based on the first observations.
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];

  // Use tiny constant for numerical stability.
  Type eps = Type(1e-8);

  // Initialize negative log likelihood.
  Type nll = 0.0;

  // 4. Loop over time steps (avoid using current response values in predictions)
  //    Equation Descriptions:
  //    [1] COTS Dynamics: Next value = current + growth_rate * current * outbreak multiplier
  //        with logistic damping based on outbreak_trigger to represent resource limitation.
  //    [2] Fast Coral Dynamics: Decline from predation + regeneration towards 100% cover.
  //    [3] Slow Coral Dynamics: Similar to fast coral with different rate parameters.
  for (int t = 1; t < n; t++) {
    // Determine outbreak multiplier using a Hill function for a smooth transition:
    // Avoid negative or zero values in the power function by using safe_cots.
    Type safe_cots = cots_pred[t-1] > eps ? cots_pred[t-1] : eps;
    Type outbreak = Type(1.0) + (outbreak_efficiency - Type(1.0)) 
                     * ( pow(safe_cots, outbreak_hill_exp) / ( pow(outbreak_trigger, outbreak_hill_exp) + pow(safe_cots, outbreak_hill_exp) ) );
    
    // [1] COTS population dynamics:
    cots_pred[t] = cots_pred[t-1] + growth_rate * cots_pred[t-1] * outbreak * (Type(1.0) - cots_pred[t-1]/(outbreak_trigger + eps));
    cots_pred[t] = cots_pred[t] < eps ? eps : cots_pred[t];
    
    // [2] Fast-growing coral dynamics:
    fast_pred[t] = fast_pred[t-1] - coral_predation_rate_fast * cots_pred[t-1] * fast_pred[t-1] 
                   + coral_regen_fast * (Type(100.0) - fast_pred[t-1]);
    fast_pred[t] = fast_pred[t] < eps ? eps : fast_pred[t];
    
    // [3] Slow-growing coral dynamics:
    slow_pred[t] = slow_pred[t-1] - coral_predation_rate_slow * cots_pred[t-1] * slow_pred[t-1] 
                   + coral_regen_slow * (Type(100.0) - slow_pred[t-1]);
    slow_pred[t] = slow_pred[t] < eps ? eps : slow_pred[t];
    
    // Likelihood calculation using lognormal error distribution.
    Type safe_cots_pred = cots_pred[t] > eps ? cots_pred[t] : eps;
    Type safe_fast_pred = fast_pred[t] > eps ? fast_pred[t] : eps;
    Type safe_slow_pred = slow_pred[t] > eps ? slow_pred[t] : eps;
    nll -= dnorm(log(cots_dat[t] + eps), log(safe_cots_pred), exp(log_sd), true);
    nll -= dnorm(log(fast_dat[t] + eps), log(safe_fast_pred), exp(log_sd), true);
    nll -= dnorm(log(slow_dat[t] + eps), log(safe_slow_pred), exp(log_sd), true);
  }

  // 5. Reporting predicted values:
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
