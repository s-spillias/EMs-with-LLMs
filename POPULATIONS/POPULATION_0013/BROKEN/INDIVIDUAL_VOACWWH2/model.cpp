#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Observation years
  DATA_VECTOR(cots_dat); // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (Acropora spp., %)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (Faviidae/Porites, %)
  DATA_VECTOR(sst_dat); // Sea Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (individuals/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (year^-1)
  PARAMETER(log_K_cots); // log COTS carrying capacity (individuals/m2)
  PARAMETER(log_m_cots); // log baseline COTS mortality rate (year^-1)
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/year)
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/year)
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral
  PARAMETER(log_r_fast); // log fast coral growth rate (year^-1)
  PARAMETER(log_r_slow); // log slow coral growth rate (year^-1)
  PARAMETER(log_K_fast); // log fast coral carrying capacity (% cover)
  PARAMETER(log_K_slow); // log slow coral carrying capacity (% cover)
  PARAMETER(log_m_fast); // log fast coral mortality (year^-1)
  PARAMETER(log_m_slow); // log slow coral mortality (year^-1)
  PARAMETER(beta_sst_cots); // SST effect on COTS growth (per deg C)
  PARAMETER(beta_sst_coral); // SST effect on coral growth (per deg C)
  PARAMETER(log_sigma_cots); // log obs SD for COTS
  PARAMETER(log_sigma_fast); // log obs SD for fast coral
  PARAMETER(log_sigma_slow); // log obs SD for slow coral

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type m_cots = exp(log_m_cots); // COTS baseline mortality
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral
  Type e_fast = exp(log_e_fast); // COTS assimilation efficiency (fast coral)
  Type e_slow = exp(log_e_slow); // COTS assimilation efficiency (slow coral)
  Type r_fast = exp(log_r_fast); // Fast coral growth rate
  Type r_slow = exp(log_r_slow); // Slow coral growth rate
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type m_fast = exp(log_m_fast); // Fast coral mortality
  Type m_slow = exp(log_m_slow); // Slow coral mortality
  Type sigma_cots = exp(log_sigma_cots); // COTS obs SD
  Type sigma_fast = exp(log_sigma_fast); // Fast coral obs SD
  Type sigma_slow = exp(log_sigma_slow); // Slow coral obs SD

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- SMALL CONSTANTS FOR NUMERICAL STABILITY ---
  Type eps = Type(1e-8);

  // --- INITIAL STATES ---
  if(n > 0) {
    cots_pred(0) = CppAD::CondExpGt(cots_dat(0), eps, cots_dat(0), eps);
    fast_pred(0) = CppAD::CondExpGt(fast_dat(0), eps, fast_dat(0), eps);
    slow_pred(0) = CppAD::CondExpGt(slow_dat(0), eps, slow_dat(0), eps);
  }

  // --- PROCESS MODEL ---
  if(n > 0) {
    for(int t=1; t<n; t++){
      // Use only previous time step predictions (not observations)
      Type cots_prev = cots_pred(t-1);
      Type fast_prev = fast_pred(t-1);
      Type slow_prev = slow_pred(t-1);

      // 1. Resource limitation for COTS (saturating, coral-dependent)
      Type coral_food = fast_prev + slow_prev + eps; // total coral cover (%)
      Type food_lim = coral_food / (coral_food + Type(10.0)); // saturating function

      // 2. SST effect on COTS growth (centered at 27C)
      Type sst_eff_cots = exp(beta_sst_cots * (sst_dat(t-1) - Type(27.0)));

      // 3. COTS predation functional response (Type II, separate for each coral group)
      Type pred_fast = alpha_fast * cots_prev * fast_prev / (fast_prev + Type(5.0) + eps); // m2/year
      Type pred_slow = alpha_slow * cots_prev * slow_prev / (slow_prev + Type(5.0) + eps);

      // 4. COTS population update
      Type cots_growth = r_cots * cots_prev * food_lim * sst_eff_cots;
      Type cots_gain = cots_growth + e_fast * pred_fast + e_slow * pred_slow + cotsimm_dat(t-1);
      Type cots_loss = m_cots * cots_prev + pred_fast + pred_slow;
      Type cots_next = cots_prev + cots_gain - cots_loss;

      // 5. Carrying capacity penalty (soft, smooth)
      Type K_penalty = exp(-pow((cots_next/K_cots), 4));
      cots_next = cots_next * K_penalty + K_cots * (Type(1.0) - K_penalty);

      // 6. Fast coral update
      Type sst_eff_fast = exp(beta_sst_coral * (sst_dat(t-1) - Type(27.0)));
      Type fast_growth = r_fast * fast_prev * (Type(1.0) - (fast_prev+slow_prev)/K_fast) * sst_eff_fast;
      Type fast_next = fast_prev + fast_growth - pred_fast - m_fast * fast_prev;

      // 7. Slow coral update
      Type sst_eff_slow = exp(beta_sst_coral * (sst_dat(t-1) - Type(27.0)));
      Type slow_growth = r_slow * slow_prev * (Type(1.0) - (fast_prev+slow_prev)/K_slow) * sst_eff_slow;
      Type slow_next = slow_prev + slow_growth - pred_slow - m_slow * slow_prev;

      // 8. Boundaries (numerical stability)
      cots_next = CppAD::CondExpGt(cots_next, eps, cots_next, eps);
      fast_next = CppAD::CondExpGt(fast_next, eps, fast_next, eps);
      slow_next = CppAD::CondExpGt(slow_next, eps, slow_next, eps);

      // 9. Save predictions
      cots_pred(t) = cots_next;
      fast_pred(t) = fast_next;
      slow_pred(t) = slow_next;
    }
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  if(n > 0) {
    for(int t=0; t<n; t++){
      // Only calculate likelihood if t < data length
      if (t < cots_dat.size() && t < fast_dat.size() && t < slow_dat.size()) {
        // Lognormal likelihood for strictly positive data
        nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots + Type(0.05), true);
        nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast + Type(0.05), true);
        nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow + Type(0.05), true);
      }
    }
  }

  // --- REPORTING ---
  ADREPORT(cots_pred); // Predicted COTS abundance (individuals/m2)
  ADREPORT(fast_pred); // Predicted fast coral cover (%)
  ADREPORT(slow_pred); // Predicted slow coral cover (%)
  REPORT(cots_pred); // Predicted COTS abundance (individuals/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  // --- EQUATION DESCRIPTIONS ---
  /*
  1. food_lim = coral_food / (coral_food + 10): Resource limitation for COTS (saturating).
  2. sst_eff_cots = exp(beta_sst_cots * (sst - 27)): SST effect on COTS growth.
  3. pred_fast = alpha_fast * cots * fast / (fast + 5): COTS predation on fast coral (Type II).
  4. pred_slow = alpha_slow * cots * slow / (slow + 5): COTS predation on slow coral (Type II).
  5. cots_next = cots + growth + assimilation + immigration - mortality - predation.
  6. K_penalty = exp(-((cots_next/K_cots)^4)): Soft carrying capacity penalty.
  7. fast_next = fast + growth - predation - mortality.
  8. slow_next = slow + growth - predation - mortality.
  9. All rates are modulated by SST and resource limitation.
  10. Lognormal likelihood for all observations.
  */

  return nll;
}
