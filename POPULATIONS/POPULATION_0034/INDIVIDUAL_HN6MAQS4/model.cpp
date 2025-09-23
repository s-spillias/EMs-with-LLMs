#include <TMB.hpp>

// Crown-of-thorns starfish outbreak dynamics model for the Great Barrier Reef

template<class Type>
Type objective_function<Type>::operator() () {
  
  // ======================
  // DATA INPUTS
  // ======================
  DATA_VECTOR(Year);               // observation years
  DATA_VECTOR(cots_dat);           // observed adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat);           // observed fast coral cover (% Acropora)
  DATA_VECTOR(slow_dat);           // observed slow coral cover (% Faviidae, Porites)
  DATA_VECTOR(sst_dat);            // observed sea surface temperatures (C)
  DATA_VECTOR(cotsimm_dat);        // observed larval immigration pulses (ind/m2/year)

  // ======================
  // PARAMETERS
  // ======================
  PARAMETER(log_r_cots);           // log intrinsic growth rate of COTS (per year)
  PARAMETER(log_K_cots);           // log carrying capacity of COTS (ind/m2)
  PARAMETER(log_mort_cots);        // log baseline mortality of adult COTS (per year)
  PARAMETER(log_attack_fast);      // log attack rate on fast coral cover (% consumed per COTS per year)
  PARAMETER(log_attack_slow);      // log attack rate on slow coral cover (% consumed per COTS per year)
  PARAMETER(log_grow_fast);        // log intrinsic recovery/growth rate of fast coral (% cover per year)
  PARAMETER(log_grow_slow);        // log intrinsic recovery/growth rate of slow coral (% cover per year)
  PARAMETER(log_K_coral);          // log total possible coral cover (%)
  PARAMETER(beta_sst);             // temperature sensitivity factor for COTS growth
  PARAMETER(log_sd_cots);          // log observation error sd for COTS
  PARAMETER(log_sd_coral);         // log observation error sd for coral

  // ======================
  // TRANSFORM PARAMETERS
  // ======================
  Type r_cots = exp(log_r_cots);         // COTS growth rate
  Type K_cots = exp(log_K_cots) + Type(1e-8);  // COTS carrying capacity
  Type mort_cots = exp(log_mort_cots);   // natural mortality
  Type attack_fast = exp(log_attack_fast); // feeding on Acropora
  Type attack_slow = exp(log_attack_slow); // feeding on Porites/Faviidae
  Type grow_fast = exp(log_grow_fast);   // coral growth rate
  Type grow_slow = exp(log_grow_slow);   // coral growth rate
  Type K_coral = exp(log_K_coral);       // total max coral cover
  Type sd_cots = exp(log_sd_cots) + Type(1e-6);   // minimum error floor
  Type sd_coral = exp(log_sd_coral) + Type(1e-6); // minimum error floor

  int n = Year.size();

  // ======================
  // STATE VECTORS
  // ======================
  vector<Type> cots_pred(n);   
  vector<Type> fast_pred(n);   
  vector<Type> slow_pred(n);   

  // ======================
  // INITIAL CONDITIONS
  // ======================
  cots_pred(0) = cots_dat(0);   // initialize at first observation
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ======================
  // PROCESS MODEL
  // ======================
  for(int t=1; t<n; t++){
    
    // ---- Coral resource dynamics (previous time step values only)
    Type total_coral_prev = fast_pred(t-1) + slow_pred(t-1);
    if (total_coral_prev > K_coral) total_coral_prev = K_coral;

    // Coral recovery with logistic competition for space
    Type fast_growth = grow_fast * fast_pred(t-1) * (1 - total_coral_prev / K_coral);
    Type slow_growth = grow_slow * slow_pred(t-1) * (1 - total_coral_prev / K_coral);

    // Coral loss to predation by COTS (functional response saturating with coral availability)
    Type consump_fast = attack_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(1e-8));
    Type consump_slow = attack_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(1e-8));
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - consump_fast;
    slow_pred(t) = slow_pred(t-1) + slow_growth - consump_slow;

    // Bound coral to [0, K_coral]
    if (fast_pred(t) < 0) fast_pred(t) = 0;
    if (slow_pred(t) < 0) slow_pred(t) = 0;
    if (fast_pred(t) > K_coral) fast_pred(t) = K_coral;
    if (slow_pred(t) > K_coral) slow_pred(t) = K_coral;

    // ---- COTS dynamics (logistic with environmental modification)
    Type env_factor = exp(beta_sst * (sst_dat(t-1) - 27.0)); // relative to baseline 27C
    Type cots_growth = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * env_factor;
    Type cots_mort = mort_cots * cots_pred(t-1);
    Type recruit = cotsimm_dat(t-1); // immigration pulses from external sources

    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mort + recruit;

    if (cots_pred(t) < 0) cots_pred(t) = 0;
  }

  // ======================
  // LIKELIHOOD
  // ======================
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihood for COTS densities
    nll -= dnorm(log(cots_dat(t)+Type(1e-8)), log(cots_pred(t)+Type(1e-8)), sd_cots, true);
    // Normal likelihood for coral cover
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_coral, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_coral, true);
  }

  // ======================
  // REPORT SECTION
  // ======================
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
