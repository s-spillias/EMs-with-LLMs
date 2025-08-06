#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. DATA SECTION
  // ---------------
  // Time variable (must match data file exactly)
  DATA_VECTOR(Year); // Observation year (integer, years)

  // Observed variables (must match _dat names in data file)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(sst_dat);     // Sea-surface temperature (deg C)
  DATA_VECTOR(cots_dat);    // Observed adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (Acropora, %)
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (Faviidae/Porites, %)

  // 2. PARAMETER SECTION
  // --------------------
  // COTS population parameters
  PARAMETER(log_r_cots);      // log intrinsic growth rate of COTS (log(year^-1))
  PARAMETER(log_K_cots);      // log carrying capacity for COTS (log(indiv/m2))
  PARAMETER(log_alpha_cots);  // log predation efficiency on coral (log(% cover)^-1 year^-1)
  PARAMETER(log_immig_eff);   // log efficiency of larval immigration (log(indiv/m2)/(indiv/m2/year))
  PARAMETER(log_m_cots);      // log background mortality rate of COTS (log(year^-1))
  PARAMETER(logit_phi_outbreak); // logit threshold for outbreak triggering (unitless, 0-1 scaled)
  PARAMETER(log_sigma_cots);  // log SD for COTS observation error (lognormal, log(indiv/m2))

  // Coral parameters
  PARAMETER(log_r_fast);      // log recovery rate of fast coral (log(year^-1))
  PARAMETER(log_r_slow);      // log recovery rate of slow coral (log(year^-1))
  PARAMETER(log_K_coral);     // log total coral carrying capacity (% cover)
  PARAMETER(log_m_fast);      // log background mortality of fast coral (log(year^-1))
  PARAMETER(log_m_slow);      // log background mortality of slow coral (log(year^-1))
  PARAMETER(log_sigma_fast);  // log SD for fast coral obs error (lognormal, log(%))
  PARAMETER(log_sigma_slow);  // log SD for slow coral obs error (lognormal, log(%))

  // Environmental effect parameters
  PARAMETER(beta_sst_cots);   // Effect of SST on COTS growth (year^-1/degC)
  PARAMETER(beta_sst_coral);  // Effect of SST on coral mortality (year^-1/degC)

  // 3. TRANSFORM PARAMETERS TO NATURAL SCALE
  // ----------------------------------------
  Type r_cots = exp(log_r_cots);           // Intrinsic COTS growth rate (year^-1)
  Type K_cots = exp(log_K_cots);           // COTS carrying capacity (indiv/m2)
  Type alpha_cots = exp(log_alpha_cots);   // COTS predation efficiency (per % coral per year)
  Type immig_eff = exp(log_immig_eff);     // Immigration efficiency (indiv/m2 per indiv/m2/year)
  Type m_cots = exp(log_m_cots);           // Background COTS mortality (year^-1)
  Type phi_outbreak = 1/(1+exp(-logit_phi_outbreak)); // Outbreak threshold (0-1, smooth)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // SD for COTS obs (lognormal, indiv/m2)

  Type r_fast = exp(log_r_fast);           // Fast coral recovery rate (year^-1)
  Type r_slow = exp(log_r_slow);           // Slow coral recovery rate (year^-1)
  Type K_coral = exp(log_K_coral);         // Total coral carrying capacity (% cover)
  Type m_fast = exp(log_m_fast);           // Fast coral background mortality (year^-1)
  Type m_slow = exp(log_m_slow);           // Slow coral background mortality (year^-1)
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // SD for fast coral obs (lognormal, %)
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // SD for slow coral obs (lognormal, %)

  // 4. INITIAL CONDITIONS
  // ---------------------
  int n = Year.size();
  vector<Type> cots_pred(n);  // Predicted COTS abundance (indiv/m2)
  vector<Type> fast_pred(n);  // Predicted fast coral cover (%)
  vector<Type> slow_pred(n);  // Predicted slow coral cover (%)

  // Set initial conditions to first observed value, but ensure strictly positive for log-likelihood
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-8), cots_dat(0), Type(1e-4));
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-4));
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-4));

  // 5. PROCESS MODEL
  // ----------------
  // Numbered equation descriptions:
  // (1) COTS population: density-dependent growth, resource limitation, immigration, SST effect, outbreak threshold
  // (2) Coral: logistic recovery, COTS predation (selective), SST effect, background mortality, competition for space

  for(int t=1; t<n; t++) {
    // Resource limitation: total coral cover available
    Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // (to avoid zero)
    Type coral_frac = coral_total_prev / (K_coral + Type(1e-8)); // Fraction of coral cover

    // Outbreak trigger: smooth threshold on immigration
    Type outbreak_trigger = 1/(1 + exp(-10*(cotsimm_dat(t-1) - phi_outbreak))); // (0-1, smooth)

    // COTS population dynamics (Eq 1)
    Type cots_growth = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/(K_cots + Type(1e-8))); // Logistic growth
    Type cots_resource = alpha_cots * cots_pred(t-1) * coral_frac; // Resource-limited predation
    Type cots_immig = immig_eff * cotsimm_dat(t-1) * outbreak_trigger; // Immigration, modulated by outbreak trigger
    Type cots_env = beta_sst_cots * (sst_dat(t-1) - 28.0); // SST effect (centered at 28C)
    Type cots_mort = m_cots * cots_pred(t-1); // Background mortality

    cots_pred(t) = cots_pred(t-1) + cots_growth + cots_immig + cots_env - cots_resource - cots_mort;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-4)); // Prevent negative and zero

    // Coral dynamics (Eq 2)
    // Fast coral (Acropora)
    Type fast_recovery = r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1))/(K_coral + Type(1e-8)));
    Type fast_predation = alpha_cots * cots_pred(t-1) * (fast_pred(t-1)/(coral_total_prev + Type(1e-8))); // Selective predation
    Type fast_env = beta_sst_coral * (sst_dat(t-1) - 28.0) * fast_pred(t-1); // SST effect
    Type fast_mort = m_fast * fast_pred(t-1);

    fast_pred(t) = fast_pred(t-1) + fast_recovery - fast_predation - fast_env - fast_mort;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-4)); // Prevent negative and zero

    // Slow coral (Faviidae/Porites)
    Type slow_recovery = r_slow * slow_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1))/(K_coral + Type(1e-8)));
    Type slow_predation = alpha_cots * cots_pred(t-1) * (slow_pred(t-1)/(coral_total_prev + Type(1e-8))); // Selective predation
    Type slow_env = beta_sst_coral * (sst_dat(t-1) - 28.0) * slow_pred(t-1); // SST effect
    Type slow_mort = m_slow * slow_pred(t-1);

    slow_pred(t) = slow_pred(t-1) + slow_recovery - slow_predation - slow_env - slow_mort;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-4)); // Prevent negative and zero
  }

  // 6. LIKELIHOOD
  // -------------
  // Use lognormal likelihoods for all strictly positive data, with minimum SD for stability
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // COTS
    nll -= dnorm(log(CppAD::CondExpGt(cots_dat(t), Type(1e-8), cots_dat(t), Type(1e-4))),
                 log(CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-4))),
                 sigma_cots, true);
    // Fast coral
    nll -= dnorm(log(CppAD::CondExpGt(fast_dat(t), Type(1e-8), fast_dat(t), Type(1e-4))),
                 log(CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-4))),
                 sigma_fast, true);
    // Slow coral
    nll -= dnorm(log(CppAD::CondExpGt(slow_dat(t), Type(1e-8), slow_dat(t), Type(1e-4))),
                 log(CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-4))),
                 sigma_slow, true);
  }

  // 7. PENALTIES FOR PARAMETER BOUNDS (smooth, not hard)
  // ----------------------------------------------------
  // Example: penalize if parameters go outside plausible biological ranges
  Type penalty = 0.0;
  penalty += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots-Type(0.01), Type(0.0)), 2); // r_cots >= 0.01
  penalty += pow(CppAD::CondExpGt(r_cots, Type(2.0), r_cots-Type(2.0), Type(0.0)), 2);   // r_cots <= 2
  penalty += pow(CppAD::CondExpLt(K_cots, Type(0.01), K_cots-Type(0.01), Type(0.0)), 2); // K_cots >= 0.01
  penalty += pow(CppAD::CondExpGt(K_cots, Type(10.0), K_cots-Type(10.0), Type(0.0)), 2); // K_cots <= 10
  // (Add more penalties as needed for other parameters)

  // 8. OBJECTIVE FUNCTION
  // ---------------------
  nll += penalty;
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  return nll;
}
