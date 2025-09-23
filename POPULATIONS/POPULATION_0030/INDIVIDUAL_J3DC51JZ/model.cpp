#include <TMB.hpp>

// Crown-of-Thorns Starfish (COTS) outbreak dynamics model
// Captures boom-bust cycles of COTS and effects on coral cover (fast- and slow-growing species)

// Template Model Builder objective function
template<class Type>
Type objective_function<Type>::operator() ()
{
  // =====================
  // 1. DATA INPUTS
  // =====================
  DATA_VECTOR(Year);                 // Time vector in years
  DATA_VECTOR(cots_dat);             // Observed adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat);             // Observed Acropora (fast-growing coral) cover (%)
  DATA_VECTOR(slow_dat);             // Observed Porites/Faviidae (slow-growing coral) cover (%)
  DATA_VECTOR(sst_dat);              // Environmental driver: sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // Larval immigration rate (ind/m2/year)

  // =====================
  // 2. PARAMETERS
  // =====================
  PARAMETER(log_r_cots);      // Intrinsic growth rate of COTS (log scale, year^-1)
  PARAMETER(log_K_cots);      // Carrying capacity of COTS determined by coral resources (log scale, ind/m2)
  PARAMETER(log_alpha_feed);  // Attack rate functional response scaling (log scale)
  PARAMETER(log_handling);    // Handling time for predation (log scale, year)
  PARAMETER(log_eff_fast);    // Feeding efficiency on fast corals (log scale)
  PARAMETER(log_eff_slow);    // Feeding efficiency on slow corals (log scale)
  PARAMETER(log_r_fast);      // Coral intrinsic regrowth rate for Acropora (%/yr, log scale)
  PARAMETER(log_r_slow);      // Coral intrinsic regrowth rate for slow corals (%/yr, log scale)
  PARAMETER(log_K_coral);     // Max coral cover (%), shared carrying capacity of coral cover (log scale)
  PARAMETER(beta_sst);        // Effect of SST anomalies on COTS growth
  PARAMETER(log_sigma_cots);  // Observation error for COTS (log scale)
  PARAMETER(log_sigma_fast);  // Observation error for fast-growing coral (%)
  PARAMETER(log_sigma_slow);  // Observation error for slow-growing coral (%)

  // =====================
  // 3. TRANSFORM PARAMETERS
  // =====================
  Type r_cots     = exp(log_r_cots);     // COTS growth rate
  Type K_cots     = exp(log_K_cots);     // COTS carrying capacity
  Type alpha_feed = exp(log_alpha_feed); // Attack rate
  Type handling   = exp(log_handling);   // Handling time
  Type eff_fast   = exp(log_eff_fast);   // Efficiency of feeding on Acropora
  Type eff_slow   = exp(log_eff_slow);   // Efficiency of feeding on Porites/Faviidae
  Type r_fast     = exp(log_r_fast);     // Fast coral regrowth
  Type r_slow     = exp(log_r_slow);     // Slow coral regrowth
  Type K_coral    = exp(log_K_coral);    // Total coral cover capacity
  Type sigma_cots = exp(log_sigma_cots); // Observation error
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);

  // =====================
  // 4. STATE VARIABLES (predictions)
  // =====================
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // =====================
  // 5. PROCESS MODEL
  // =====================
  for(int t=1; t<n; t++) {
    // Total coral cover constraint
    Type total_coral_prev = fast_pred(t-1) + slow_pred(t-1);
    Type crowding_effect = (K_coral - total_coral_prev) / (K_coral + Type(1e-8));

    // COTS growth with logistic and SST effect + immigration
    Type growth_term = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots);
    Type sst_effect = beta_sst * (sst_dat(t-1) - 27.0); // deviation from baseline 27°C
    Type immigration = cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t-1) + growth_term * exp(sst_effect) + immigration;

    // Functional response to coral prey (Holling type II)
    Type prey_fast = fast_pred(t-1);
    Type prey_slow = slow_pred(t-1);
    Type intake_fast = alpha_feed * prey_fast / (1 + alpha_feed * handling * prey_fast);
    Type intake_slow = alpha_feed * prey_slow / (1 + alpha_feed * handling * prey_slow);

    // Coral depletion by predation
    Type pred_loss_fast = eff_fast * cots_pred(t-1) * intake_fast;
    Type pred_loss_slow = eff_slow * cots_pred(t-1) * intake_slow;

    // Coral dynamics with logistic growth and predation
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * crowding_effect - pred_loss_fast;
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * crowding_effect - pred_loss_slow;

    // Numerical stability bounds
    if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);
    if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
    if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);
  }

  // =====================
  // 6. LIKELIHOOD
  // =====================
  Type nll = 0;
  for(int t=0; t<n; t++) {
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);
  }

  // =====================
  // 7. REPORT
  // =====================
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
