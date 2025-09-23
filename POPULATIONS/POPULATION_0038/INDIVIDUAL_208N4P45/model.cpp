#include <TMB.hpp>

// Crown-of-thorns starfish (COTS) - Coral Dynamics Model
// Emphasis: episodic outbreaks, boom–bust dynamics, trophic interactions,
//           influence of environmental forcing (SST, immigration pulses)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // === DATA INPUTS ===
  DATA_VECTOR(Year);                  // Time vector (years)
  DATA_VECTOR(cots_dat);              // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%) Acropora
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%) Porites + Faviidae
  DATA_VECTOR(sst_dat);               // Sea Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);           // Immigration rate (individuals/m2/year)

  int n = Year.size();

  // === PARAMETERS ===
  PARAMETER(log_r_cots);              // Intrinsic COTS population growth rate (year^-1)
  PARAMETER(log_K_cots);              // Carrying capacity of COTS (individuals/m2)
  PARAMETER(log_alpha_fast);          // Attack rate on fast coral (% cover consumed per individual per year)
  PARAMETER(log_alpha_slow);          // Attack rate on slow coral (% cover consumed per individual per year)
  PARAMETER(log_eff_cots);            // Assimilation efficiency of consumed coral (% converted to COTS growth)
  PARAMETER(log_r_fast);              // Intrinsic growth rate of fast coral (%/year)
  PARAMETER(log_K_fast);              // Carrying capacity for fast coral (% cover)
  PARAMETER(log_r_slow);              // Intrinsic growth rate of slow coral (%/year)
  PARAMETER(log_K_slow);              // Carrying capacity for slow coral (% cover)
  PARAMETER(beta_sst_cots);           // Temperature sensitivity of COTS growth
  PARAMETER(beta_sst_coral);          // Temperature sensitivity of coral growth
  PARAMETER(log_sigma_cots);          // Observation error for COTS
  PARAMETER(log_sigma_fast);          // Observation error for fast coral
  PARAMETER(log_sigma_slow);          // Observation error for slow coral

  // === TRANSFORM PARAMETERS TO NATURAL SCALE ===
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_fast = exp(log_alpha_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type eff_cots = exp(log_eff_cots);
  Type r_fast = exp(log_r_fast);
  Type K_fast = exp(log_K_fast);
  Type r_slow = exp(log_r_slow);
  Type K_slow = exp(log_K_slow);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);

  // === STATE VECTORS FOR PREDICTIONS ===
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // === INITIAL CONDITIONS (set to first observation) ===
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // === PROCESS MODEL ===
  for(int t = 1; t < n; t++){
    // Previous state values
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);

    // 1. COTS population growth with logistic density dependence and SST modulation
    Type sst_effect = exp(beta_sst_cots * (sst_dat(t-1) - 27.0));  // baseline temperature effect
    Type cots_growth = r_cots * cots_prev * (1.0 - cots_prev / (K_cots + Type(1e-8))) * sst_effect;

    // 2. Coral consumption functional response (Type II saturating response, Holling's)
    Type cons_fast = alpha_fast * cots_prev * fast_prev / (1.0 + alpha_fast * fast_prev + Type(1e-8));
    Type cons_slow = alpha_slow * cots_prev * slow_prev / (1.0 + alpha_slow * slow_prev + Type(1e-8));

    // 3. Net COTS update: growth + assimilation from feeding + immigration pulses
    Type cots_new = cots_prev + cots_growth + eff_cots * (cons_fast + cons_slow) + cotsimm_dat(t-1);
    if(cots_new < Type(1e-8)) cots_new = Type(1e-8);
    cots_pred(t) = cots_new;

    // 4. Coral dynamics with logistic growth, consumption loss, and SST sensitivity
    Type fast_growth = r_fast * fast_prev * (1.0 - fast_prev / (K_fast + Type(1e-8))) * exp(beta_sst_coral * (sst_dat(t-1)-27.0));
    Type slow_growth = r_slow * slow_prev * (1.0 - slow_prev / (K_slow + Type(1e-8))) * exp(beta_sst_coral * (sst_dat(t-1)-27.0));

    Type fast_new = fast_prev + fast_growth - cons_fast;
    Type slow_new = slow_prev + slow_growth - cons_slow;

    if(fast_new < Type(1e-8)) fast_new = Type(1e-8);
    if(slow_new < Type(1e-8)) slow_new = Type(1e-8);

    fast_pred(t) = fast_new;
    slow_pred(t) = slow_new;
  }

  // === LIKELIHOOD ===
  Type nll = 0.0;

  // Use lognormal likelihood for positive data
  for(int t=0; t<n; t++){
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
  }

  // === REPORTS ===
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equations:
1. dCOTS/dt = logistic growth + SST forcing + assimilation of coral consumption + immigration
2. Coral consumption = Holling Type II functional response
3. Coral dynamics = logistic growth * SST modulation – consumption loss
4. Likelihood = lognormal fit between predicted vs observed values
*/
