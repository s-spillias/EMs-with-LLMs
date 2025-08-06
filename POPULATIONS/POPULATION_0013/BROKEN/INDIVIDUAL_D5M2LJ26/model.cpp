#include <TMB.hpp>

// 1. COTS = Crown-of-Thorns starfish (individuals/m^2)
// 2. fast = Fast-growing coral (Acropora spp.) (% cover)
// 3. slow = Slow-growing coral (Faviidae/Porites) (% cover)
// 4. sst = Sea-surface temperature (deg C)
// 5. cotsimm = COTS larval immigration (individuals/m^2/year)
// 6. All _dat variables are observed; _pred are model predictions

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time (years)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (individuals/m^2/year)

  int n = Year.size();

  // Defensive check for empty data
  if(n == 0) return Type(0.0);

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS recruitment rate (log(year^-1))
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m^2))
  PARAMETER(log_m_cots); // log COTS natural mortality (log(year^-1))
  PARAMETER(log_alpha_pred); // log max COTS predation rate on coral (log(% cover/year))
  PARAMETER(log_beta_fast); // log COTS preference for fast coral (logit-scale)
  PARAMETER(log_beta_slow); // log COTS preference for slow coral (logit-scale)
  PARAMETER(log_effic_pred); // log efficiency of coral-to-COTS conversion (log(unitless))
  PARAMETER(log_r_fast); // log fast coral growth rate (log(year^-1))
  PARAMETER(log_r_slow); // log slow coral growth rate (log(year^-1))
  PARAMETER(log_K_fast); // log fast coral carrying capacity (log(% cover))
  PARAMETER(log_K_slow); // log slow coral carrying capacity (log(% cover))
  PARAMETER(log_m_fast); // log fast coral natural mortality (log(year^-1))
  PARAMETER(log_m_slow); // log slow coral natural mortality (log(year^-1))
  PARAMETER(logit_thresh_outbreak); // logit threshold for COTS outbreak (logit(indiv/m^2))
  PARAMETER(log_sigma_cots); // log obs SD for COTS (log(indiv/m^2))
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (log(% cover))
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (log(% cover))
  PARAMETER(logit_sst_sens); // logit sensitivity of COTS recruitment to SST (logit(unitless))
  PARAMETER(logit_immig_eff); // logit efficiency of COTS immigration (logit(unitless))
  PARAMETER(log_h_coral); // log half-saturation for coral-dependent COTS recruitment (log(% cover))

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS recruitment rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m^2)
  Type m_cots = exp(log_m_cots); // COTS mortality (year^-1)
  Type alpha_pred = exp(log_alpha_pred); // Max predation rate (% cover/year)
  Type beta_fast = 1/(1+exp(-log_beta_fast)); // Preference for fast coral (0-1)
  Type beta_slow = 1/(1+exp(-log_beta_slow)); // Preference for slow coral (0-1)
  Type effic_pred = exp(log_effic_pred); // Coral-to-COTS conversion efficiency
  Type r_fast = exp(log_r_fast); // Fast coral growth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (year^-1)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (% cover)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (% cover)
  Type m_fast = exp(log_m_fast); // Fast coral mortality (year^-1)
  Type m_slow = exp(log_m_slow); // Slow coral mortality (year^-1)
  Type thresh_outbreak = 1/(1+exp(-logit_thresh_outbreak)); // Outbreak threshold (0-1 scaled to K_cots)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // SD for COTS
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // SD for fast coral
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // SD for slow coral
  Type sst_sens = 1/(1+exp(-logit_sst_sens)); // SST sensitivity (0-1)
  Type immig_eff = 1/(1+exp(-logit_immig_eff)); // Immigration efficiency (0-1)
  Type h_coral = exp(log_h_coral); // Half-saturation for coral-dependent COTS recruitment

  // --- INITIAL STATES ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-8), cots_dat(0), Type(1e-8)); // Initial COTS from data, prevent log(0)
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-8)); // Initial fast coral from data, prevent log(0)
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-8)); // Initial slow coral from data, prevent log(0)

  // --- MODEL DYNAMICS ---
  for(int t=1; t<n; t++) {
    // 1. Coral predation pressure (saturating functional response)
    Type coral_avail = beta_fast * fast_pred(t-1) + beta_slow * slow_pred(t-1) + Type(1e-8); // Weighted coral cover
    Type pred_rate = alpha_pred * cots_pred(t-1) * coral_avail / (coral_avail + Type(10.0)); // Saturating predation

    // 2. COTS recruitment (modulated by SST, coral availability, and immigration)
    Type sst_effect = 1 + sst_sens * (sst_dat(t-1) - Type(27.0)); // SST effect (centered at 27C)
    Type immig = immig_eff * cotsimm_dat(t-1); // Immigration effect

    // Coral feedback on COTS recruitment (resource limitation)
    Type coral_feedback = coral_avail / (coral_avail + h_coral + Type(1e-8));

    // 3. Outbreak threshold effect (smooth, not hard)
    Type outbreak_mod = 1/(1+exp(-10*(cots_pred(t-1)/K_cots - thresh_outbreak))); // Smooth threshold

    // 4. COTS population update
    Type recruit = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * sst_effect * coral_feedback * outbreak_mod + immig;
    Type pred_gain = effic_pred * pred_rate; // Biomass gain from predation
    Type cots_next = cots_pred(t-1) + recruit + pred_gain - m_cots * cots_pred(t-1);
    // Defensive: handle NaN/Inf for AD types (cannot use R_FINITE)
    if(!(cots_next == cots_next) || cots_next <= Type(0)) cots_next = Type(1e-8);
    cots_pred(t) = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8)); // Prevent negative

    // 5. Fast coral update (logistic growth minus predation)
    Type fast_growth = r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast);
    Type fast_loss = pred_rate * (beta_fast * fast_pred(t-1) / (coral_avail + Type(1e-8)));
    Type fast_next = fast_pred(t-1) + fast_growth - fast_loss - m_fast * fast_pred(t-1);
    if(!(fast_next == fast_next) || fast_next <= Type(0)) fast_next = Type(1e-8);
    fast_pred(t) = CppAD::CondExpGt(fast_next, Type(1e-8), fast_next, Type(1e-8));

    // 6. Slow coral update (logistic growth minus predation)
    Type slow_growth = r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow);
    Type slow_loss = pred_rate * (beta_slow * slow_pred(t-1) / (coral_avail + Type(1e-8)));
    Type slow_next = slow_pred(t-1) + slow_growth - slow_loss - m_slow * slow_pred(t-1);
    if(!(slow_next == slow_next) || slow_next <= Type(0)) slow_next = Type(1e-8);
    slow_pred(t) = CppAD::CondExpGt(slow_next, Type(1e-8), slow_next, Type(1e-8));
  }

  // --- LIKELIHOOD ---
  Type nll = 0;
  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive data
    // Defensive: avoid log(0) or negative values in likelihood
    nll -= dnorm(log(CppAD::CondExpGt(cots_dat(t), Type(1e-8), cots_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8))),
                 sigma_cots, true);
    nll -= dnorm(log(CppAD::CondExpGt(fast_dat(t), Type(1e-8), fast_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8))),
                 sigma_fast, true);
    nll -= dnorm(log(CppAD::CondExpGt(slow_dat(t), Type(1e-8), slow_dat(t), Type(1e-8))),
                 log(CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8))),
                 sigma_slow, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // --- EQUATION DESCRIPTIONS ---
  /*
    1. coral_avail = beta_fast * fast_pred + beta_slow * slow_pred
       (Weighted coral cover available to COTS)
    2. pred_rate = alpha_pred * cots_pred * coral_avail / (coral_avail + 10)
       (Saturating COTS predation on coral)
    3. sst_effect = 1 + sst_sens * (sst - 27)
       (SST modifies COTS recruitment)
    4. outbreak_mod = 1/(1+exp(-10*(cots_pred/K_cots - thresh_outbreak)))
       (Smooth outbreak threshold effect)
    5. recruit = r_cots * cots_pred * (1 - cots_pred/K_cots) * sst_effect * outbreak_mod + immig
       (COTS recruitment, density-dependent, SST and outbreak modulated)
    6. pred_gain = effic_pred * pred_rate
       (COTS gain from coral predation)
    7. cots_next = cots_pred + recruit + pred_gain - m_cots * cots_pred
       (COTS population update)
    8. fast_next = fast_pred + r_fast * fast_pred * (1 - fast_pred/K_fast) - fast_loss - m_fast * fast_pred
       (Fast coral update)
    9. slow_next = slow_pred + r_slow * slow_pred * (1 - slow_pred/K_slow) - slow_loss - m_slow * slow_pred
       (Slow coral update)
  */

  return nll;
}
