#include <TMB.hpp>

// Model for episodic outbreaks of Crown-of-Thorns starfish (COTS) and their impacts on coral communities

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. DATA INPUTS
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (indiv/m2/year)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (Acropora spp.) (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (Faviidae/Porites spp.) (%)

  int n = Year.size();

  // 2. PARAMETERS

  // COTS population parameters
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (indiv/m2)
  PARAMETER(log_m_cots); // log natural mortality rate of COTS (year^-1)
  PARAMETER(log_alpha_pred); // log attack rate of COTS on coral (%^-1 year^-1)
  PARAMETER(log_beta_fast); // log preference for fast coral (unitless)
  PARAMETER(log_beta_slow); // log preference for slow coral (unitless)
  PARAMETER(log_eff_cots); // log conversion efficiency of coral to COTS biomass (unitless)
  PARAMETER(log_immig_eff); // log efficiency of larval immigration (unitless)
  PARAMETER(log_thresh_coral); // log coral cover threshold for outbreak (%, smooth threshold)
  PARAMETER(log_sigma_cots); // log SD of COTS obs error (lognormal)
  
  // Coral parameters
  PARAMETER(log_r_fast); // log growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity of fast coral (%)
  PARAMETER(log_K_slow); // log carrying capacity of slow coral (%)
  PARAMETER(log_m_fast); // log natural mortality of fast coral (year^-1)
  PARAMETER(log_m_slow); // log natural mortality of slow coral (year^-1)
  PARAMETER(log_sigma_fast); // log SD of fast coral obs error (lognormal)
  PARAMETER(log_sigma_slow); // log SD of slow coral obs error (lognormal)

  // Environmental effect parameters
  PARAMETER(beta_sst_cots); // effect of SST on COTS growth (per deg C)
  PARAMETER(beta_sst_coral); // effect of SST on coral growth (per deg C)

  // 3. TRANSFORM PARAMETERS
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type m_cots = exp(log_m_cots); // COTS natural mortality (year^-1)
  Type alpha_pred = exp(log_alpha_pred); // COTS attack rate on coral (%^-1 year^-1)
  Type beta_fast = exp(log_beta_fast); // COTS preference for fast coral
  Type beta_slow = exp(log_beta_slow); // COTS preference for slow coral
  Type eff_cots = exp(log_eff_cots); // COTS conversion efficiency
  Type immig_eff = exp(log_immig_eff); // Immigration efficiency
  Type thresh_coral = exp(log_thresh_coral); // Coral cover threshold for outbreak (%)
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs error

  Type r_fast = exp(log_r_fast); // Fast coral growth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (year^-1)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type m_fast = exp(log_m_fast); // Fast coral natural mortality (year^-1)
  Type m_slow = exp(log_m_slow); // Slow coral natural mortality (year^-1)
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs error
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs error

  // 4. INITIAL STATES (use first obs as initial condition)
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  cots_pred(0) = cots_dat(0) + Type(1e-8); // Initial COTS abundance (indiv/m2)
  fast_pred(0) = fast_dat(0) + Type(1e-8); // Initial fast coral cover (%)
  slow_pred(0) = slow_dat(0) + Type(1e-8); // Initial slow coral cover (%)

  // 5. PROCESS MODEL
  for(int t=1; t<n; t++) {

    // 5.1. Coral availability for COTS predation (weighted sum)
    Type coral_avail = beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1) + Type(1e-8); // % coral cover available

    // 5.2. Outbreak trigger: smooth threshold on coral cover
    Type outbreak_factor = 1.0 / (1.0 + exp(-(coral_avail - thresh_coral + Type(1e-8)))); // Smooth transition (0-1) as coral_avail exceeds threshold

    // 5.3. Immigration pulse (modulated by efficiency)
    Type immig = immig_eff * cotsimm_dat(t) * outbreak_factor;

    // 5.4. COTS predation functional response (Holling Type II, saturating)
    Type pred_rate = alpha_pred * coral_avail / (coral_avail + Type(1.0)); // %^-1 year^-1

    // 5.5. COTS population update
    // Growth is density-dependent, modulated by SST and resource limitation
    Type env_cots = exp(beta_sst_cots * (sst_dat(t) - Type(27.0))); // SST effect (centered at 27C)
    Type resource_lim = coral_avail / (coral_avail + Type(10.0)); // Resource limitation (saturating)
    Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) * resource_lim * env_cots;

    // Total predation loss (COTS feeding on coral)
    Type coral_loss = pred_rate * cots_pred(t-1);

    // COTS update
    cots_pred(t) = cots_pred(t-1)
      + cots_growth // population growth
      + immig // larval immigration
      - m_cots * cots_pred(t-1) // natural mortality
      - coral_loss * (1.0 - eff_cots); // loss due to predation inefficiency

    // Prevent negative values
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8));

    // 5.6. Coral updates (fast and slow)
    // Fast coral
    Type env_fast = exp(beta_sst_coral * (sst_dat(t) - Type(27.0)));
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/K_fast) * env_fast;
    Type fast_pred_loss = pred_rate * cots_pred(t-1) * (beta_fast * fast_pred(t-1) / (coral_avail + Type(1e-8)));
    fast_pred(t) = fast_pred(t-1)
      + fast_growth
      - m_fast * fast_pred(t-1)
      - fast_pred_loss;

    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8));

    // Slow coral
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/K_slow) * env_fast;
    Type slow_pred_loss = pred_rate * cots_pred(t-1) * (beta_slow * slow_pred(t-1) / (coral_avail + Type(1e-8)));
    slow_pred(t) = slow_pred(t-1)
      + slow_growth
      - m_slow * slow_pred(t-1)
      - slow_pred_loss;

    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8));
    // Defensive: set to small value if not finite (TMB/CppAD safe)
    // Only check for NaN, not Inf (Inf is rare in TMB/CppAD and NaN is the main risk for segfaults)
    // Remove all NaN checks: TMB/CppAD does not guarantee ==NaN works, and this can itself cause segfaults on some platforms.
    // Instead, clamp all state variables to a minimum value after each update.
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8));
  }

  // 6. LIKELIHOOD (lognormal, fixed minimum SD)
  Type nll = 0.0;
  Type minSD = Type(1e-3);

  for(int t=0; t<n; t++) {
    // COTS
    Type sd_cots = sqrt(sigma_cots*sigma_cots + minSD*minSD);
    Type cots_pred_log = log(cots_pred(t) + Type(1e-8));
    if(cots_pred_log < Type(-20.0)) cots_pred_log = Type(-20.0); // Clamp to avoid -Inf
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), cots_pred_log, sd_cots, true);

    // Fast coral
    Type sd_fast = sqrt(sigma_fast*sigma_fast + minSD*minSD);
    Type fast_pred_log = log(fast_pred(t) + Type(1e-8));
    if(fast_pred_log < Type(-20.0)) fast_pred_log = Type(-20.0);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), fast_pred_log, sd_fast, true);

    // Slow coral
    Type sd_slow = sqrt(sigma_slow*sigma_slow + minSD*minSD);
    Type slow_pred_log = log(slow_pred(t) + Type(1e-8));
    if(slow_pred_log < Type(-20.0)) slow_pred_log = Type(-20.0);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), slow_pred_log, sd_slow, true);
  }

  // 7. SMOOTH PENALTIES FOR PARAMETER BOUNDS (example: keep rates positive and within plausible ranges)
  // (Penalties can be adjusted as needed for biological realism)
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots-Type(0.01), Type(0.0)), 2); // r_cots >= 0.01
  nll += pow(CppAD::CondExpGt(r_cots, Type(5.0), r_cots-Type(5.0), Type(0.0)), 2); // r_cots <= 5

  // 8. REPORTING
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // 9. EQUATION DESCRIPTIONS
  /*
    1. COTS population: 
       cots_pred(t) = cots_pred(t-1) + growth + immigration - mortality - predation loss
    2. Coral availability: 
       coral_avail = beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1)
    3. Outbreak trigger: 
       outbreak_factor = 1/(1+exp(-(coral_avail-thresh_coral)))
    4. Immigration: 
       immig = immig_eff * cotsimm_dat(t) * outbreak_factor
    5. COTS predation: 
       pred_rate = alpha_pred * coral_avail / (coral_avail + 1)
    6. Fast coral: 
       fast_pred(t) = fast_pred(t-1) + growth - mortality - predation
    7. Slow coral: 
       slow_pred(t) = slow_pred(t-1) + growth - mortality - predation
  */

  return nll;
}
