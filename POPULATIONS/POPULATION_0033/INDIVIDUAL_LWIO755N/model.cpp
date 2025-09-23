#include <TMB.hpp>

// Template Model Builder (TMB) model for Crown-of-thorns starfish (COTS) outbreaks
// Captures boom-bust cycles and selective predation on coral communities

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ----------------------
  // DATA INPUTS
  // ----------------------
  DATA_VECTOR(Year);          // Time steps (years)
  DATA_VECTOR(cots_dat);      // Observed adult COTS abundance (ind/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (ind/m2/year)

  int n = Year.size();

  // ----------------------
  // MODEL PARAMETERS
  // ----------------------
  PARAMETER(log_r_cots);        // Intrinsic growth rate of COTS (log scale, year^-1)
  PARAMETER(log_K_cots);        // Carrying capacity of COTS (log scale, ind/m2)
  PARAMETER(log_m_cots);        // Natural mortality rate of COTS (log scale, year^-1)
  PARAMETER(log_alpha_fast);    // Attack rate on fast corals (log scale)
  PARAMETER(log_alpha_slow);    // Attack rate on slow corals (log scale)
  PARAMETER(log_h);             // Handling time for feeding (log scale, years)
  PARAMETER(beta_temp);         // Sensitivity of COTS growth to SST deviations (per degree C)
  PARAMETER(log_g_fast);        // Growth rate of fast corals (log scale, %/year)
  PARAMETER(log_g_slow);        // Growth rate of slow corals (log scale, %/year)
  PARAMETER(log_K_coral);       // Coral carrying capacity (% cover)
  PARAMETER(log_sigma_cots);    // Observation error sd for COTS (log scale)
  PARAMETER(log_sigma_coral);   // Observation error sd for coral (%)

  // ----------------------
  // TRANSFORM PARAMETERS
  // ----------------------
  Type r_cots = exp(log_r_cots);        // COTS growth rate
  Type K_cots = exp(log_K_cots);        // Carrying capacity of COTS
  Type m_cots = exp(log_m_cots);        // Natural mortality rate
  Type alpha_fast = exp(log_alpha_fast);// Attack rate fast-growing coral
  Type alpha_slow = exp(log_alpha_slow);// Attack rate slow-growing coral
  Type h = exp(log_h);                  // Handling time
  Type g_fast = exp(log_g_fast);        // Growth rate fast coral
  Type g_slow = exp(log_g_slow);        // Growth rate slow coral
  Type K_coral = exp(log_K_coral);      // Max coral cover
  Type sigma_cots = exp(log_sigma_cots);// Obs error COTS
  Type sigma_coral = exp(log_sigma_coral);// Obs error corals

  // ----------------------
  // STATE VECTORS
  // ----------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // ----------------------
  // INITIAL CONDITIONS
  // ----------------------
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ----------------------
  // PROCESS MODEL
  // ----------------------
  for(int t=1; t<n; t++){
    // Previous state
    Type Ct = cots_pred(t-1);
    Type F = fast_pred(t-1);
    Type S = slow_pred(t-1);

    // 1. Coral mortality due to COTS (Type II functional response with handling time)
    Type consumption_fast = (alpha_fast * Ct * F) / (1.0 + h * alpha_fast * F + Type(1e-8));
    Type consumption_slow = (alpha_slow * Ct * S) / (1.0 + h * alpha_slow * S + Type(1e-8));

    // 2. Coral dynamics (logistic regrowth minus COTS predation)
    Type F_next = F + g_fast * F * (1 - (F + S)/K_coral) - consumption_fast;
    Type S_next = S + g_slow * S * (1 - (F + S)/K_coral) - consumption_slow;
    F_next = CppAD::CondExpGt(F_next, Type(0.0), F_next, Type(1e-8));
    S_next = CppAD::CondExpGt(S_next, Type(0.0), S_next, Type(1e-8));

    // 3. Environmental effect on COTS growth
    Type temp_effect = exp(beta_temp * (sst_dat(t) - 27.0)); // baseline SST ~27 °C

    // 4. COTS dynamics (logistic growth, immigration, predation feedback via coral availability)
    Type coral_food = F + 0.5 * S; // fast coral preferred, weight slow by 0.5
    Type food_effect = coral_food / (coral_food + Type(1.0));
    Type Ct_next = Ct + r_cots * Ct * (1 - Ct/K_cots) * temp_effect * food_effect
                   - m_cots * Ct
                   + cotsimm_dat(t);

    Ct_next = CppAD::CondExpGt(Ct_next, Type(0.0), Ct_next, Type(1e-8));

    // Store predictions
    cots_pred(t) = Ct_next;
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
  }

  // ----------------------
  // LIKELIHOOD
  // ----------------------
  Type nll = 0.0;

  for(int t=0; t<n; t++){
    // Lognormal likelihood for COTS (strictly positive)
    nll -= dnorm(log(cots_dat(t)+1e-8), log(cots_pred(t)+1e-8), sigma_cots, true);
    // Normal likelihood for coral % cover
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_coral, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_coral, true);
  }

  // ----------------------
  // REPORT
  // ----------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equation summary:
1. Consumption_fast = (α_fast * COTS * Fast_corals)/(1 + h * α_fast * Fast_corals) : Holling Type II functional response
2. Consumption_slow = (α_slow * COTS * Slow_corals)/(1 + h * α_slow * Slow_corals)
3. Coral growth = logistic growth - COTS predation
4. COTS growth = logistic growth * SST effect * Coral availability - mortality + immigration
*/
