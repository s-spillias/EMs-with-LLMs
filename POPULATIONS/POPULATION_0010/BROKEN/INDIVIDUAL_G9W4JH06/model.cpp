#include <TMB.hpp>

// 1. COTS = Crown-of-Thorns Starfish
// 2. fast = Fast-growing coral (Acropora spp.)
// 3. slow = Slow-growing coral (Faviidae, Porites spp.)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/yr)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow coral cover (%)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (log(yr^-1))
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m2))
  PARAMETER(log_alpha_fast); // log predation rate on fast coral (log(%^-1 yr^-1))
  PARAMETER(log_alpha_slow); // log predation rate on slow coral (log(%^-1 yr^-1))
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (log(unitless))
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (log(unitless))
  PARAMETER(log_r_fast); // log fast coral regrowth rate (log(yr^-1))
  PARAMETER(log_r_slow); // log slow coral regrowth rate (log(yr^-1))
  PARAMETER(log_K_fast); // log fast coral carrying capacity (log(% cover))
  PARAMETER(log_K_slow); // log slow coral carrying capacity (log(% cover))
  PARAMETER(log_sigma_cots); // log obs SD for COTS (log(indiv/m2))
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (log(%))
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (log(%))
  PARAMETER(beta_sst); // effect of SST on COTS recruitment (unitless)
  PARAMETER(log_sst_opt); // log optimal SST for COTS recruitment (log(deg C))
  PARAMETER(log_sst_sd); // log SD of SST effect (log(deg C))
  PARAMETER(log_immig_scale); // log scaling for larval immigration (log(indiv/m2/yr))

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // Intrinsic COTS growth rate (yr^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_fast = exp(log_alpha_fast); // Predation rate on fast coral (%^-1 yr^-1)
  Type alpha_slow = exp(log_alpha_slow); // Predation rate on slow coral (%^-1 yr^-1)
  Type e_fast = exp(log_e_fast); // Assimilation efficiency from fast coral (unitless)
  Type e_slow = exp(log_e_slow); // Assimilation efficiency from slow coral (unitless)
  Type r_fast = exp(log_r_fast); // Fast coral regrowth rate (yr^-1)
  Type r_slow = exp(log_r_slow); // Slow coral regrowth rate (yr^-1)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (% cover)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (% cover)
  Type sigma_cots = exp(log_sigma_cots); // Obs SD for COTS (indiv/m2)
  Type sigma_fast = exp(log_sigma_fast); // Obs SD for fast coral (%)
  Type sigma_slow = exp(log_sigma_slow); // Obs SD for slow coral (%)
  Type sst_opt = exp(log_sst_opt); // Optimal SST for COTS recruitment (deg C)
  Type sst_sd = exp(log_sst_sd); // SD of SST effect (deg C)
  Type immig_scale = exp(log_immig_scale); // Scaling for larval immigration (indiv/m2/yr)

  // --- INITIAL STATES ---
  Type cots = cots_dat(0); // Initial COTS abundance (indiv/m2)
  Type fast = fast_dat(0); // Initial fast coral cover (%)
  Type slow = slow_dat(0); // Initial slow coral cover (%)

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- SMALL CONSTANTS FOR NUMERICAL STABILITY ---
  Type eps = Type(1e-8);

  // --- MODEL DYNAMICS ---
  for(int t=0; t<n; t++) {
    // 1. Save predictions
    cots_pred(t) = cots;
    fast_pred(t) = fast;
    slow_pred(t) = slow;

    // 2. Calculate environmental effect on COTS recruitment (Gaussian function of SST)
    Type sst_effect = exp(-pow(sst_dat(t) - sst_opt, 2) / (2 * pow(sst_sd, 2) + eps));
    // 3. Immigration input (scaled)
    Type immig = immig_scale * cotsimm_dat(t);

    // 4. Resource limitation: total available coral (fast + slow)
    Type coral_avail = fast + slow + eps;

    // 5. COTS population update (discrete-time, resource-limited, with immigration and environmental effect)
    Type predation = alpha_fast * fast + alpha_slow * slow; // Total predation rate
    Type resource_lim = coral_avail / (coral_avail + Type(10.0)); // Saturating function for resource limitation
    Type cots_growth = r_cots * cots * resource_lim * sst_effect; // Growth modulated by resources and SST
    Type cots_next = cots + cots_growth - predation * cots + immig; // Update with predation loss and immigration

    // 6. Bound COTS abundance to [0, K_cots] smoothly
    cots_next = K_cots * (cots_next / (K_cots + cots_next + eps));

    // 7. Coral updates (fast and slow), with regrowth, predation, and competition
    // Fast coral
    Type fast_regrow = r_fast * fast * (1 - (fast + slow) / (K_fast + K_slow + eps)); // Regrowth, limited by total coral
    Type fast_pred_loss = alpha_fast * fast * cots * e_fast; // Loss to COTS predation
    Type fast_next = fast + fast_regrow - fast_pred_loss;
    fast_next = CppAD::CondExpGt(fast_next, Type(0.0), fast_next, eps); // Bound to >=0

    // Slow coral
    Type slow_regrow = r_slow * slow * (1 - (fast + slow) / (K_fast + K_slow + eps));
    Type slow_pred_loss = alpha_slow * slow * cots * e_slow;
    Type slow_next = slow + slow_regrow - slow_pred_loss;
    slow_next = CppAD::CondExpGt(slow_next, Type(0.0), slow_next, eps);

    // 8. Advance state (for next time step)
    if(t < n-1) {
      // Defensive: ensure state variables are not NaN or negative
      cots = (cots_next == cots_next) ? cots_next : Type(0.0); // if NaN, set to 0
      fast = (fast_next == fast_next) ? fast_next : Type(0.0);
      slow = (slow_next == slow_next) ? slow_next : Type(0.0);
      // Also ensure non-negativity
      cots = CppAD::CondExpGt(cots, Type(0.0), cots, eps);
      fast = CppAD::CondExpGt(fast, Type(0.0), fast, eps);
      slow = CppAD::CondExpGt(slow, Type(0.0), slow, eps);
    }
  }

  // --- LIKELIHOOD CALCULATION ---
  Type nll = Type(0.0);

  // 1. COTS: lognormal likelihood (strictly positive, boom-bust cycles)
  for(int t=0; t<n; t++) {
    Type pred = cots_pred(t) + eps;
    Type obs = cots_dat(t) + eps;
    Type sd = sigma_cots + Type(0.05); // Minimum SD for stability
    nll -= dnorm(log(obs), log(pred), sd, true);
  }

  // 2. Fast coral: lognormal likelihood
  for(int t=0; t<n; t++) {
    Type pred = fast_pred(t) + eps;
    Type obs = fast_dat(t) + eps;
    Type sd = sigma_fast + Type(0.05);
    nll -= dnorm(log(obs), log(pred), sd, true);
  }

  // 3. Slow coral: lognormal likelihood
  for(int t=0; t<n; t++) {
    Type pred = slow_pred(t) + eps;
    Type obs = slow_dat(t) + eps;
    Type sd = sigma_slow + Type(0.05);
    nll -= dnorm(log(obs), log(pred), sd, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  // --- EQUATION DESCRIPTIONS ---
  // 1. COTS_next = K_cots * (cots + r_cots * cots * resource_lim * sst_effect - predation * cots + immig) / (K_cots + cots + r_cots * cots * resource_lim * sst_effect - predation * cots + immig)
  //    (COTS population update with resource limitation, SST effect, predation, and immigration; bounded by K_cots)
  // 2. fast_next = max(fast + r_fast * fast * (1 - (fast + slow)/(K_fast + K_slow)) - alpha_fast * fast * cots * e_fast, eps)
  //    (Fast coral regrowth minus COTS predation loss; bounded >=0)
  // 3. slow_next = max(slow + r_slow * slow * (1 - (fast + slow)/(K_fast + K_slow)) - alpha_slow * slow * cots * e_slow, eps)
  //    (Slow coral regrowth minus COTS predation loss; bounded >=0)
  // 4. sst_effect = exp(-((sst_dat - sst_opt)^2) / (2*sst_sd^2))
  //    (Gaussian effect of SST on COTS recruitment)
  // 5. resource_lim = (fast + slow) / (fast + slow + 10)
  //    (Saturating function for resource limitation)

  return nll;
}
