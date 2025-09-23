#include <TMB.hpp>

// TEMPLATE MODEL BUILDER (TMB) model
// Modeling Crown-of-Thorns Starfish (COTS) outbreaks on the Great Barrier Reef
// Emphasizes capturing timing, magnitude, and duration of boom-bust cycles, 
// including coral functional groups (fast vs slow corals).

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -------------------------
  // 1. DATA INPUTS
  // -------------------------

  DATA_VECTOR(Year);                         // Time variable (years)
  DATA_VECTOR(sst_dat);                      // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);                  // COTS larval immigration (individuals/m2/year)
  DATA_VECTOR(cots_dat);                     // Observed adult COTS (individuals/m2)
  DATA_VECTOR(fast_dat);                     // Observed fast coral (Acropora, % cover)
  DATA_VECTOR(slow_dat);                     // Observed slow coral (Faviidae, Porites, % cover)

  // -------------------------
  // 2. PARAMETERS
  // -------------------------
  
  PARAMETER(log_r_cots);               // Intrinsic COTS growth rate (on log scale, year^-1)
  PARAMETER(log_K_cots);               // Carrying capacity of COTS (log scale, ind/m2)
  PARAMETER(log_alpha_pred);           // Feeding rate of COTS on Acropora (%/year per ind/m2)
  PARAMETER(log_beta_pred);            // Feeding rate of COTS on slow corals (%/year per ind/m2)
  PARAMETER(log_r_fast);               // Growth rate of fast corals (% cover/year)
  PARAMETER(log_r_slow);               // Growth rate of slow corals (% cover/year)
  PARAMETER(log_K_coral);              // Coral carrying capacity (% cover)
  PARAMETER(env_sst_effect);           // Effect of SST anomalies on COTS recruitment
  PARAMETER(sd_proc);                  // Process error SD
  PARAMETER(sd_obs);                   // Observation error SD
  
  // Transform parameters back to natural scale where required
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_pred = exp(log_alpha_pred);
  Type beta_pred = exp(log_beta_pred);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type sigma_proc = exp(sd_proc);
  Type sigma_obs = exp(sd_obs);

  int n = Year.size();

  // -------------------------
  // 3. STATES & PREDICTIONS
  // -------------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialization: use first data as initial condition
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // -------------------------
  // 4. DYNAMIC EQUATIONS
  // -------------------------
  for (int t = 1; t < n; t++) {
    
    // Environmental boost to larval success (smooth exponential scaling to SST and immigration)
    Type env_effect = exp(env_sst_effect * (sst_dat(t-1) - Type(27.0))) + Type(1e-8);
    Type larval_input = cotsimm_dat(t-1) * env_effect;
    
    // 4.1 COTS dynamics: logistic with larval input and feedback from corals (resource availability)
    Type coral_food = fast_pred(t-1) + slow_pred(t-1);
    Type food_effect = coral_food / (Type(5.0) + coral_food); // saturating response
    
    cots_pred(t) = cots_pred(t-1) + 
                   r_cots * cots_pred(t-1) * (1 - (cots_pred(t-1)/K_cots)) * food_effect
                   + larval_input;
    
    // Avoid negatives
    if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);

    // 4.2 Coral dynamics
    Type total_coral_prev = fast_pred(t-1) + slow_pred(t-1);
    
    // Fast coral (Acropora), heavily eaten by COTS
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1 - (total_coral_prev/K_coral)) 
                   - alpha_pred * cots_pred(t-1) * fast_pred(t-1);
    if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);

    // Slow coral (Porites, Faviidae), less preferred by COTS
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1 - (total_coral_prev/K_coral))
                   - beta_pred * cots_pred(t-1) * slow_pred(t-1);
    if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
  }

  // -------------------------
  // 5. LIKELIHOOD
  // -------------------------
  Type nll = 0.0;

  for(int t = 0; t < n; t++){
    // Observation likelihoods (lognormal for strictly positive data)
    nll -= dnorm(log(cots_dat(t)+Type(1e-8)),
                 log(cots_pred(t)+Type(1e-8)),
                 sigma_obs,
                 true);
    nll -= dnorm(log(fast_dat(t)+Type(1e-8)),
                 log(fast_pred(t)+Type(1e-8)),
                 sigma_obs,
                 true);
    nll -= dnorm(log(slow_dat(t)+Type(1e-8)),
                 log(slow_pred(t)+Type(1e-8)),
                 sigma_obs,
                 true);
  }

  // -------------------------
  // 6. REPORTING
  // -------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equation summary:
1. COTS dynamics: Logistic growth with larval immigration input and saturating food feedback
2. Coral dynamics: Logistic growth toward carrying capacity, minus COTS predation losses
3. Environmental forcing: SST influences larval settlement success
4. Feedback loops: Coral depletion reduces COTS persistence, driving bust phase
5. Predictions matched to observed time series via lognormal likelihood
*/
