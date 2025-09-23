#include <TMB.hpp>

// Crown-of-thorns starfish outbreak model
// Predicts boom-bust dynamics and coral cover under predation pressure

template<class Type>
Type objective_function<Type>::operator() ()
{
  // === DATA INPUTS ===
  DATA_VECTOR(Year);           // Time in years
  DATA_VECTOR(sst_dat);        // Sea-Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // Larval immigration rate (individuals/m²/year)
  DATA_VECTOR(cots_dat);       // Observed adult COTS abundance (individuals/m²)
  DATA_VECTOR(fast_dat);       // Observed fast coral cover (%) - Acropora
  DATA_VECTOR(slow_dat);       // Observed slow coral cover (%) - Porites, Faviidae

  // === MODEL PARAMETERS ===
  PARAMETER(log_r_cots);    // Intrinsic growth rate of COTS (log scale)
  PARAMETER(log_K_cots);    // Carrying capacity for COTS (log scale)
  PARAMETER(log_alpha_pred);// Predation rate on corals (log scale)
  PARAMETER(log_pref_fast); // Preferential predation weight for fast corals (log scale)
  PARAMETER(log_pref_slow); // Preferential predation weight for slow corals (log scale)
  PARAMETER(log_recov_fast);// Recovery rate of fast corals (log scale)
  PARAMETER(log_recov_slow);// Recovery rate of slow corals (log scale)
  PARAMETER(log_m_base);    // Background mortality of COTS (log scale)
  PARAMETER(beta_temp);     // Temperature effect on COTS recruitment
  PARAMETER(log_sd_cots);   // Observation error SD for COTS (log scale)
  PARAMETER(log_sd_coral);  // Observation error SD for coral cover (log scale)

  // === TRANSFORMED PARAMETERS ===
  Type r_cots    = exp(log_r_cots);     // Growth rate (year^-1)
  Type K_cots    = exp(log_K_cots);     // Max COTS density (ind/m²)
  Type alpha_pred= exp(log_alpha_pred); // Attack rate on coral (% cover per ind)
  Type pref_fast = exp(log_pref_fast);  // Weight of feeding preference on fast coral
  Type pref_slow = exp(log_pref_slow);  // Weight of feeding preference on slow coral
  Type recov_fast= exp(log_recov_fast); // Recovery rate fast corals (% per yr)
  Type recov_slow= exp(log_recov_slow); // Recovery rate slow corals (% per yr)
  Type m_base    = exp(log_m_base);     // Background COTS mortality (year^-1)
  Type sd_cots   = exp(log_sd_cots) + Type(1e-6);
  Type sd_coral  = exp(log_sd_coral) + Type(1e-6);

  int n = Year.size();

  // === STATE VARIABLES ===
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // INITIAL CONDITIONS from first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // === PROCESS MODEL ===
  for(int t=1; t<n; t++){
    // 1. Recruitment pulse: modified by immigration & SST
    Type recruit = cotsimm_dat(t-1) * exp(beta_temp * (sst_dat(t-1)-27.0));

    // 2. Population growth: logistic form with mortality
    Type cots_growth = cots_pred(t-1) + r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots);
    Type cots_mortality = m_base * cots_pred(t-1);

    // Net COTS abundance this year
    cots_pred(t) = cots_growth + recruit - cots_mortality;
    if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8);

    // 3. Coral predation functional response (smooth saturating)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8);
    Type frac_fast   = fast_pred(t-1) / total_coral;
    Type frac_slow   = slow_pred(t-1) / total_coral;

    Type coral_consumed = alpha_pred * cots_pred(t) / (1 + alpha_pred * cots_pred(t));
    Type fast_loss = coral_consumed * pref_fast * frac_fast * fast_pred(t-1);
    Type slow_loss = coral_consumed * pref_slow * frac_slow * slow_pred(t-1);

    // 4. Coral dynamics with recovery
    fast_pred(t) = fast_pred(t-1) - fast_loss + recov_fast * (100 - fast_pred(t-1) - slow_pred(t-1));
    slow_pred(t) = slow_pred(t-1) - slow_loss + recov_slow * (100 - fast_pred(t-1) - slow_pred(t-1));

    // Bound at small positive values
    if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);
    if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
  }

  // === LIKELIHOOD COMPONENTS ===
  Type nll = 0.0;

  for(int t=0; t<n; t++){
    nll -= dnorm(log(cots_dat(t)+1e-6), log(cots_pred(t)+1e-6), sd_cots, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_coral, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_coral, true);
  }

  // === REPORTS ===
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equation summary:
1. recruit = immigration × exp(beta_temp × (SST - 27))
2. cots_pred[t] = cots_pred[t-1] + r_cots * cots_pred[t-1]*(1 - cots_pred[t-1]/K_cots) + recruit - mortality
3. coral_consumed = (alpha_pred * COTS) / (1 + alpha_pred * COTS)  [saturating response]
4. fast_pred[t] = fast_pred[t-1] - fast_loss + recov_fast*(free_space)
5. slow_pred[t] = slow_pred[t-1] - slow_loss + recov_slow*(free_space)
*/
