#include <TMB.hpp>

// Crown-of-Thorns Starfish Outbreak Model
// Captures episodic boom-bust population cycles and selective coral predation
// Units are per year timesteps

template<class Type>
Type objective_function<Type>::operator() () {
  DATA_VECTOR(Year); // Time index
  DATA_VECTOR(sst_dat); // Sea-surface temperature forcing (°C)
  DATA_VECTOR(cotsimm_dat); // Immigration rate (larvae / m2 / yr)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (ind/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%)

  // Parameters ----------------------------
  PARAMETER(log_r_cots);    // ln intrinsic growth rate of COTS (1/yr)
  PARAMETER(log_m_cots);    // ln natural mortality rate of COTS (1/yr)
  PARAMETER(log_attack_fast); // ln max feeding rate on Acropora (%/yr per ind)
  PARAMETER(log_attack_slow); // ln max feeding rate on slow corals (%/yr per ind)
  PARAMETER(log_handling);    // ln handling time parameter (yr/%)
  PARAMETER(log_K_fast);    // ln carrying capacity of Acropora (% cover)
  PARAMETER(log_K_slow);    // ln carrying capacity of slow corals (% cover)
  PARAMETER(log_temp_sens); // ln temperature sensitivity scaling
  PARAMETER(log_sd_obs);    // ln observational error SD

  // Transform parameters for stability
  Type r_cots = exp(log_r_cots);    
  Type m_cots = exp(log_m_cots);    
  Type attack_fast = exp(log_attack_fast);  
  Type attack_slow = exp(log_attack_slow);  
  Type handling = exp(log_handling);    
  Type K_fast = exp(log_K_fast);   
  Type K_slow = exp(log_K_slow);   
  Type temp_sens = exp(log_temp_sens);  
  Type sd_obs = exp(log_sd_obs) + Type(1e-6);   

  int n = Year.size();

  // Predictions ----------------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize with observed initial states
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Dynamic equations ----------------------------
  for(int t=1; t<n; t++){
    // 1. Environmental influence on reproduction
    Type temp_effect = exp(temp_sens * (sst_dat(t-1) - Type(27.0))); // reference ~27°C

    // 2. COTS growth (larval immigration + survivors)
    Type recruit = r_cots * cots_pred(t-1) * temp_effect + cotsimm_dat(t-1);

    // 3. Coral consumption - Holling type II functional response
    Type food_fast = attack_fast * fast_pred(t-1) / (Type(1.0) + handling * fast_pred(t-1));
    Type food_slow = attack_slow * slow_pred(t-1) / (Type(1.0) + handling * slow_pred(t-1));
    Type total_consumption = (food_fast + food_slow) * cots_pred(t-1);

    // 4. Population update equations
    cots_pred(t) = cots_pred(t-1) + recruit - m_cots * cots_pred(t-1); 
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8));

    fast_pred(t) = fast_pred(t-1) + ( (1 - fast_pred(t-1)/K_fast) * (K_fast - fast_pred(t-1)) ) - food_fast * cots_pred(t-1);
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8));

    slow_pred(t) = slow_pred(t-1) + ( (1 - slow_pred(t-1)/K_slow) * (K_slow - slow_pred(t-1)) ) - food_slow * cots_pred(t-1);
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8));
  }

  // Likelihood ----------------------------
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihoods for strictly positive data
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sd_obs, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sd_obs, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sd_obs, true);
  }

  // Reports ----------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(r_cots);
  REPORT(m_cots);
  REPORT(attack_fast);
  REPORT(attack_slow);

  return nll;
}

/* 
Equations:
1. Reproductive recruitment of COTS is temperature-modulated.
2. Corals experience logistic-type growth limited by carrying capacity.
3. Coral losses are driven by Holling-II predation from COTS.
4. COTS experience natural mortality plus density-dependent feeding benefits.
*/
