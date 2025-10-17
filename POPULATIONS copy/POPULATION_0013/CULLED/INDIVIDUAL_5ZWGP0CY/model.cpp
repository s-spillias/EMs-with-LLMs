#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  
  // Response variables
  DATA_VECTOR(cots_dat);      // Observed Crown-of-thorns starfish abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral (Acropora spp.) cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral (Faviidae spp. and Porities spp.) cover (%)
  
  // Forcing variables
  DATA_VECTOR(sst_dat);       // Observed Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // Observed COTS larval immigration rate (individuals/m^2/year)
  
  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  
  // COTS dynamics
  PARAMETER(r_cots);          // COTS reproductive efficiency (individuals/% cover eaten)
  PARAMETER(P_max);           // Maximum coral consumption rate per COTS (% cover/individual/year)
  PARAMETER(K_cots_coral);    // Half-saturation constant for COTS predation on total coral (% cover)
  PARAMETER(m_cots);          // COTS natural mortality rate (year^-1)
  PARAMETER(pref_fast);       // COTS feeding preference for fast-growing corals (dimensionless, 0-1)
  PARAMETER(T_opt_cots);      // Optimal temperature for COTS reproduction (Celsius)
  PARAMETER(T_sig_cots);      // Standard deviation of the thermal tolerance curve for COTS reproduction (Celsius)
  
  // Coral dynamics
  PARAMETER(r_fast);          // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(r_slow);          // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(K_coral);         // Total carrying capacity for coral cover (%)
  PARAMETER(comp_fs);         // Competitive effect of slow-growing coral on fast-growing coral (dimensionless)
  PARAMETER(comp_sf);         // Competitive effect of fast-growing coral on slow-growing coral (dimensionless)
  
  // Observation error
  PARAMETER(log_sd_cots);     // Log of the standard deviation for COTS abundance
  PARAMETER(log_sd_fast);     // Log of the standard deviation for fast-growing coral cover
  PARAMETER(log_sd_slow);     // Log of the standard deviation for slow-growing coral cover
  
  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------
  
  int n_steps = cots_dat.size(); // Number of time steps in the data
  Type nll = 0.0; // Initialize negative log-likelihood
  
  // Create vectors to store model predictions
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);
  
  // Initialize predictions at time 0 with the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // ------------------------------------------------------------------------
  // ECOLOGICAL PROCESS MODEL
  //
  // Equation descriptions:
  // 1. Total coral cover at previous time step (t-1).
  // 2. Total coral consumption by COTS population (Holling Type II functional response).
  // 3. Partitioning of total consumption into fast and slow corals based on preference and availability.
  // 4. Temperature effect on COTS reproduction (Gaussian function).
  // 5. COTS population change (reproduction - mortality + immigration).
  // 6. Fast-growing coral population change (logistic growth - competition - COTS predation).
  // 7. Slow-growing coral population change (logistic growth - competition - COTS predation).
  // ------------------------------------------------------------------------
  
  for (int t = 1; t < n_steps; ++t) {
    // --- Calculate intermediate quantities at time t-1 ---
    
    // 1. Total available coral cover
    Type total_coral_tm1 = fast_pred(t-1) + slow_pred(t-1);
    
    // 2. Total coral consumption by COTS (Holling Type II)
    Type total_consumption = P_max * cots_pred(t-1) * total_coral_tm1 / (K_cots_coral + total_coral_tm1 + Type(1e-8));
    
    // 3. Partition consumption based on preference and availability
    Type fast_preference_term = pref_fast * fast_pred(t-1);
    Type slow_preference_term = (Type(1.0) - pref_fast) * slow_pred(t-1);
    Type preference_denominator = fast_preference_term + slow_preference_term + Type(1e-8);
    
    Type consumption_fast = total_consumption * (fast_preference_term / preference_denominator);
    Type consumption_slow = total_consumption * (slow_preference_term / preference_denominator);
    
    // 4. Temperature effect on COTS reproduction
    Type temp_diff = sst_dat(t-1) - T_opt_cots;
    Type temp_effect = exp(Type(-0.5) * pow(temp_diff / T_sig_cots, 2));
    
    // --- Update state variables using Euler integration ---
    
    // 5. COTS dynamics
    Type cots_reproduction = r_cots * total_consumption * temp_effect;
    Type cots_mortality = m_cots * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_reproduction - cots_mortality + cotsimm_dat(t-1);
    
    // 6. Fast-growing coral dynamics
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + comp_fs * slow_pred(t-1)) / (K_coral + Type(1e-8)));
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    
    // 7. Slow-growing coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - (slow_pred(t-1) + comp_sf * fast_pred(t-1)) / (K_coral + Type(1e-8)));
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    
    // --- Numerical stability: ensure predictions are non-negative ---
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
  }
  
  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  Type sd_cots = exp(log_sd_cots) + Type(1e-8); // Add small constant for stability
  Type sd_fast = exp(log_sd_fast) + Type(1e-8);
  Type sd_slow = exp(log_sd_slow) + Type(1e-8);
  
  for (int t = 0; t < n_steps; ++t) {
    // Lognormal likelihood for strictly positive data
    // dnorm(log(obs), log(pred), sd, log=true) is equivalent to dlnorm(obs, log(pred), sd, log=true)
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sd_cots, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sd_fast, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sd_slow, true);
  }
  
  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
