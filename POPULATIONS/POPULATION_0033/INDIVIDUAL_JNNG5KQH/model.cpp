#include <TMB.hpp>

// Crown of Thorns Outbreak Dynamics Model in TMB
// Predicts boom-bust cycles of COTS and selective predation on coral communities

template<class Type>
Type objective_function<Type>::operator() () {
  
  // =========================
  //  DATA INPUTS
  // =========================
  DATA_VECTOR(Year);              // year time series
  DATA_VECTOR(cots_dat);          // observed adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat);          // observed fast coral (Acropora cover, %)
  DATA_VECTOR(slow_dat);          // observed slow coral (Faviidae/Porites cover, %)
  DATA_VECTOR(sst_dat);           // observed sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);       // observed larval immigration input (ind/m2/year)
  
  // Number of time steps
  int n = Year.size();
  
  // =========================
  //  PARAMETERS
  // =========================
  PARAMETER(log_r_cots);          // intrinsic growth rate of COTS (log scale, year^-1)
  PARAMETER(log_K_cots);          // carrying capacity scaling for COTS (log scale, ind/m2)
  PARAMETER(log_alpha_fast);      // attack rate on fast coral (log scale, %^-1 * year^-1)
  PARAMETER(log_alpha_slow);      // attack rate on slow coral (log scale, %^-1 * year^-1)
  PARAMETER(log_h);               // half saturation constant for feeding (log scale, % coral cover)
  PARAMETER(log_m_cots);          // baseline COTS mortality (log scale, year^-1)
  PARAMETER(beta_sst);            // effect of SST anomaly on COTS survival (unitless)
  PARAMETER(log_phi);             // process error (log SD)

  // Initial condition parameters (avoid data leakage from observations)
  PARAMETER(init_cots);           // Initial value for COTS density (ind/m2, log scale)
  PARAMETER(init_fast);           // Initial cover for fast coral (%, log scale)
  PARAMETER(init_slow);           // Initial cover for slow coral (%, log scale)
  
  // Observation error parameters
  PARAMETER(log_obs_sigma_cots);  // observation error of COTS (log SD)
  PARAMETER(log_obs_sigma_fast);  // observation error of fast coral (log SD)
  PARAMETER(log_obs_sigma_slow);  // observation error of slow coral (log SD)
  
  // =========================
  //  TRANSFORM PARAMETERS
  // =========================
  Type r_cots = exp(log_r_cots); 
  Type K_cots = exp(log_K_cots); 
  Type alpha_fast = exp(log_alpha_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type h = exp(log_h);
  Type m_cots = exp(log_m_cots);
  Type phi = exp(log_phi);
  
  Type obs_sigma_cots = exp(log_obs_sigma_cots);
  Type obs_sigma_fast = exp(log_obs_sigma_fast);
  Type obs_sigma_slow = exp(log_obs_sigma_slow);
  
  // Avoid division by zero with small constant
  Type tiny = Type(1e-8);

  // Precompute mean SST (since mean() is not available)
  Type mean_sst = 0.0;
  for(int i=0; i<n; i++){
    mean_sst += sst_dat(i);
  }
  mean_sst /= n;
  
  // =========================
  //  STATE VARIABLES
  // =========================
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // =========================
  //  PROCESS MODEL
  // =========================
  // Initial conditions as explicit prediction equations
  cots_pred(0) = exp(init_cots);   // Initial predicted COTS density
  fast_pred(0) = exp(init_fast);   // Initial predicted fast coral cover
  slow_pred(0) = exp(init_slow);   // Initial predicted slow coral cover

  // Dynamic predictions
  for(int t=1; t<n; t++){
    
    // Temperature effect (centered by mean SST)
    Type sst_effect = Type(1.0) + beta_sst * (sst_dat(t-1) - mean_sst);
    
    // Coral-dependent feeding functional response (Holling type II)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + tiny;
    Type feeding_rate_fast = (alpha_fast * fast_pred(t-1)) / (h + total_coral);
    Type feeding_rate_slow = (alpha_slow * slow_pred(t-1)) / (h + total_coral);
    
    // COTS population dynamics (logistic + immigration + environment-modified mortality)
    Type growth_cots = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    Type mortality_cots = m_cots * sst_effect * cots_pred(t-1);
    Type immigration = cotsimm_dat(t-1);
    
    cots_pred(t) = cots_pred(t-1) + growth_cots - mortality_cots + immigration;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), tiny, cots_pred(t), tiny); // enforce positivity
    
    // Coral dynamics (grazing mortality from COTS + slow background recovery)
    fast_pred(t) = fast_pred(t-1) - feeding_rate_fast * cots_pred(t-1) + Type(0.05) * (Type(100.0) - fast_pred(t-1)) / Type(100.0);
    slow_pred(t) = slow_pred(t-1) - feeding_rate_slow * cots_pred(t-1) + Type(0.01) * (Type(100.0) - slow_pred(t-1)) / Type(100.0);
    
    // enforce positivity and bounds [0,100] for corals
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), tiny, fast_pred(t), tiny);
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(100.0), fast_pred(t), Type(100.0));
    
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), tiny, slow_pred(t), tiny);
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(100.0), slow_pred(t), Type(100.0));
  }
  
  // =========================
  //  LIKELIHOOD
  // =========================
  Type nll = 0.0;
  
  for(int t=0; t<n; t++){
    nll -= dnorm(log(cots_dat(t) + tiny), log(cots_pred(t) + tiny), obs_sigma_cots, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), obs_sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), obs_sigma_slow, true);
  }
  
  // =========================
  //  REPORTING
  // =========================
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);
  
  return nll;
}

/*
EQUATION SUMMARY:
1. Initial conditions are explicit predictions: cots_pred(0), fast_pred(0), slow_pred(0).
2. COTS growth: logistic growth with carrying capacity + immigration - mortality.
3. Mortality is modified by SST via a smooth multiplicative effect.
4. Feeding on corals follows a Holling type II functional response.
5. Fast coral dynamics: losses from COTS feeding + background recovery.
6. Slow coral dynamics: losses from COTS feeding + background recovery.
*/
