#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (indiv/m2/year)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (Acropora, %)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (Faviidae/Porites, %)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  // Determines the baseline rate at which COTS population increases in absence of limitation
  PARAMETER(log_K_cots); // log carrying capacity for COTS (indiv/m2)
  // Maximum sustainable COTS density, limited by coral resources
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/indiv/year)
  // COTS predation rate on Acropora
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/indiv/year)
  // COTS predation rate on Faviidae/Porites
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  // Fraction of consumed Acropora converted to COTS biomass
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  // Fraction of consumed slow coral converted to COTS biomass
  PARAMETER(log_m_cots); // log baseline COTS mortality rate (year^-1)
  // Natural mortality of adult COTS
  PARAMETER(log_immig_scale); // log scaling of larval immigration effect (indiv/m2/year)
  // Converts larval immigration to effective recruitment
  PARAMETER(log_sigma_cots); // log SD of COTS observation error (lognormal)
  PARAMETER(log_sigma_fast); // log SD of fast coral observation error (lognormal)
  PARAMETER(log_sigma_slow); // log SD of slow coral observation error (lognormal)
  PARAMETER(beta_sst); // Effect of SST anomaly on COTS growth (per deg C)
  // Modifies COTS growth rate by temperature anomaly
  PARAMETER(log_r_fast); // log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity for fast coral (% cover)
  PARAMETER(log_K_slow); // log carrying capacity for slow coral (% cover)
  PARAMETER(log_m_fast); // log baseline mortality of fast coral (year^-1)
  PARAMETER(log_m_slow); // log baseline mortality of slow coral (year^-1)
  PARAMETER(log_min_sd); // log minimum SD for likelihoods (for numerical stability)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral
  Type e_fast = exp(log_e_fast); // Assimilation efficiency from fast coral
  Type e_slow = exp(log_e_slow); // Assimilation efficiency from slow coral
  Type m_cots = exp(log_m_cots); // COTS mortality rate
  Type immig_scale = exp(log_immig_scale); // Immigration scaling
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs error
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs error
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs error
  Type r_fast = exp(log_r_fast); // Fast coral intrinsic growth
  Type r_slow = exp(log_r_slow); // Slow coral intrinsic growth
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type m_fast = exp(log_m_fast); // Fast coral mortality
  Type m_slow = exp(log_m_slow); // Slow coral mortality
  Type min_sd = exp(log_min_sd); // Minimum SD for likelihoods

  // --- INITIAL STATES ---
  PARAMETER(log_cots_0); // log initial COTS abundance (indiv/m2)
  PARAMETER(log_fast_0); // log initial fast coral cover (%)
  PARAMETER(log_slow_0); // log initial slow coral cover (%)
  Type cots_prev = exp(log_cots_0);
  Type fast_prev = exp(log_fast_0);
  Type slow_prev = exp(log_slow_0);

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- MODEL DYNAMICS ---
  // Compute mean SST for anomaly calculation
  Type sst_sum = 0.0;
  for(int t=0; t<n; t++) {
    sst_sum += sst_dat(t);
  }
  Type sst_mean = sst_sum / Type(n);

  for(int t=0; t<n; t++) {
    // 1. Environmental effect: SST anomaly (relative to mean)
    Type sst_anom = sst_dat(t) - sst_mean;

    // 2. Immigration: episodic larval supply
    Type immig = immig_scale * cotsimm_dat(t);

    // 3. Coral resource limitation (saturating functional response)
    Type coral_total = fast_prev + slow_prev + Type(1e-8); // total available coral
    Type resource_lim = coral_total / (coral_total + Type(5.0)); // saturating, threshold at ~5% cover

    // 4. COTS predation on corals (Type II functional response)
    Type pred_fast = alpha_fast * cots_prev * fast_prev / (fast_prev + Type(1.0)); // Acropora
    Type pred_slow = alpha_slow * cots_prev * slow_prev / (slow_prev + Type(1.0)); // Faviidae/Porites

    // 5. COTS population update (discrete-time, resource-limited, with environmental modifier)
    Type growth_mod = exp(beta_sst * sst_anom); // SST effect on growth
    Type cots_growth = r_cots * cots_prev * (1.0 - cots_prev / (K_cots * resource_lim)) * growth_mod;
    Type cots_gain = cots_growth + immig + e_fast * pred_fast + e_slow * pred_slow;
    Type cots_loss = m_cots * cots_prev;
    Type cots_next = cots_prev + cots_gain - cots_loss;
    cots_next = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8)); // prevent negative/zero

    // 6. Coral updates (logistic growth minus COTS predation and mortality)
    Type fast_growth = r_fast * fast_prev * (1.0 - fast_prev / (K_fast + Type(1e-8)));
    Type fast_loss = pred_fast + m_fast * fast_prev;
    Type fast_next = fast_prev + fast_growth - fast_loss;
    fast_next = CppAD::CondExpGt(fast_next, Type(1e-8), fast_next, Type(1e-8));

    Type slow_growth = r_slow * slow_prev * (1.0 - slow_prev / (K_slow + Type(1e-8)));
    Type slow_loss = pred_slow + m_slow * slow_prev;
    Type slow_next = slow_prev + slow_growth - slow_loss;
    slow_next = CppAD::CondExpGt(slow_next, Type(1e-8), slow_next, Type(1e-8));

    // 7. Save predictions
    // Model predictions for observed variables (no data leakage: only use previous time step states)
    // These lines ensure the model equations for cots_pred, fast_pred, and slow_pred are always present
    cots_pred(t) = cots_prev;   // Prediction for cots_dat (before update)
    fast_pred(t) = fast_prev;   // Prediction for fast_dat (before update)
    slow_pred(t) = slow_prev;   // Prediction for slow_dat (before update)

    // 8. Update for next time step
    cots_prev = cots_next;
    fast_prev = fast_next;
    slow_prev = slow_next;
  }

  // Ensure predictions are set for all observed variables, even if n==0
  if(n == 0) {
    cots_pred.setZero();
    fast_pred.setZero();
    slow_pred.setZero();
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;

  // Helper function for square (since not available by default)
  auto square = [](Type x) { return x * x; };

  for(int t=0; t<n; t++) {
    // Use lognormal likelihoods for strictly positive data
    Type sd_cots = sqrt(square(sigma_cots) + square(min_sd));
    Type sd_fast = sqrt(square(sigma_fast) + square(min_sd));
    Type sd_slow = sqrt(square(sigma_slow) + square(min_sd));
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sd_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sd_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sd_slow, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}

/*
Model equations (numbered):

1. sst_anom = sst_dat(t) - mean(sst_dat)
   (Sea-surface temperature anomaly, modifies COTS growth rate)

2. immig = immig_scale * cotsimm_dat(t)
   (Episodic larval immigration, triggers outbreaks)

3. resource_lim = (fast_prev + slow_prev) / ((fast_prev + slow_prev) + 5)
   (Saturating resource limitation, threshold at ~5% total coral cover)

4. pred_fast = alpha_fast * cots_prev * fast_prev / (fast_prev + 1)
   (COTS predation on Acropora, Type II functional response)

5. pred_slow = alpha_slow * cots_prev * slow_prev / (slow_prev + 1)
   (COTS predation on Faviidae/Porites, Type II functional response)

6. cots_growth = r_cots * cots_prev * (1 - cots_prev / (K_cots * resource_lim)) * exp(beta_sst * sst_anom)
   (Resource-limited, environmentally-modified COTS growth)

7. cots_gain = cots_growth + immig + e_fast * pred_fast + e_slow * pred_slow
   (Total COTS gains: growth, immigration, and assimilation from predation)

8. cots_loss = m_cots * cots_prev
   (COTS mortality)

9. cots_next = cots_prev + cots_gain - cots_loss
   (Update for next time step, bounded below by small constant)

10. fast_growth = r_fast * fast_prev * (1 - fast_prev / K_fast)
    (Logistic growth of Acropora)

11. fast_loss = pred_fast + m_fast * fast_prev
    (Losses from COTS predation and natural mortality)

12. fast_next = fast_prev + fast_growth - fast_loss
    (Update for next time step, bounded below by small constant)

13. slow_growth = r_slow * slow_prev * (1 - slow_prev / K_slow)
    (Logistic growth of slow coral)

14. slow_loss = pred_slow + m_slow * slow_prev
    (Losses from COTS predation and natural mortality)

15. slow_next = slow_prev + slow_growth - slow_loss
    (Update for next time step, bounded below by small constant)
*/
