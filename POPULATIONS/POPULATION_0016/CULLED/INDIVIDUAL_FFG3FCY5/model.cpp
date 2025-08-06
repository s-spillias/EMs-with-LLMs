#include <TMB.hpp>

// 1. Model equations describe the coupled dynamics of COTS, fast coral, and slow coral.
// 2. Resource limitation is modeled with saturating and threshold functions.
// 3. Environmental drivers (SST, larval immigration) modulate COTS recruitment.
// 4. Feedbacks: COTS reduce coral, coral depletion limits COTS, coral recovers after COTS decline.
// 5. All _pred variables are reported and correspond to _dat observations.
// 6. No current time step values of _dat variables are used in predictions (no data leakage).

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Observation year
  DATA_VECTOR(cots_dat); // Adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Fast coral cover (%)
  DATA_VECTOR(slow_dat); // Slow coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = Year.size();

  // --- GUARD AGAINST EMPTY INPUT ---
  if(n == 0) {
    // Return large penalty if no data
    return Type(1e10);
  }

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS recruitment rate (year^-1)
  PARAMETER(log_K_cots); // log COTS carrying capacity (indiv/m2)
  PARAMETER(log_alpha_fast); // log COTS predation rate on fast coral (m2/indiv/year)
  PARAMETER(log_alpha_slow); // log COTS predation rate on slow coral (m2/indiv/year)
  PARAMETER(log_r_fast); // log fast coral regrowth rate (year^-1)
  PARAMETER(log_r_slow); // log slow coral regrowth rate (year^-1)
  PARAMETER(log_K_fast); // log fast coral max cover (%)
  PARAMETER(log_K_slow); // log slow coral max cover (%)
  PARAMETER(log_beta_sst); // log SST effect on COTS recruitment (unitless)
  PARAMETER(log_imm_eff); // log efficiency of larval immigration (unitless)
  PARAMETER(log_outbreak_thresh); // log coral cover threshold for COTS outbreak (estimated)
  PARAMETER(log_outbreak_sharpness); // log sharpness of outbreak threshold (steepness of sigmoid)
  PARAMETER(log_sigma_cots); // log obs SD for COTS (lognormal)
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (lognormal)
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (lognormal)
  PARAMETER(log_K_coral_half); // log half-saturation constant for coral effect on COTS K

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // Intrinsic COTS recruitment rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_fast = exp(log_alpha_fast); // COTS predation rate on fast coral (m2/indiv/year)
  Type alpha_slow = exp(log_alpha_slow); // COTS predation rate on slow coral (m2/indiv/year)
  Type r_fast = exp(log_r_fast); // Fast coral regrowth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral regrowth rate (year^-1)
  Type K_fast = exp(log_K_fast); // Fast coral max cover (%)
  Type K_slow = exp(log_K_slow); // Slow coral max cover (%)
  Type beta_sst = exp(log_beta_sst); // SST effect on COTS recruitment (unitless)
  Type imm_eff = exp(log_imm_eff); // Efficiency of larval immigration (unitless)
  Type outbreak_thresh = exp(log_outbreak_thresh); // Coral cover threshold for COTS outbreak (estimated)
  Type outbreak_sharpness = exp(log_outbreak_sharpness); // Sharpness of outbreak threshold (steepness of sigmoid)
  Type sigma_cots = exp(log_sigma_cots); // Obs SD for COTS (lognormal)
  Type sigma_fast = exp(log_sigma_fast); // Obs SD for fast coral (lognormal)
  Type sigma_slow = exp(log_sigma_slow); // Obs SD for slow coral (lognormal)
  Type K_coral_half = exp(log_K_coral_half); // Half-saturation constant for coral effect on COTS K

  // --- INITIAL STATES ---
  Type cots_prev = cots_dat(0); // Initial COTS abundance (indiv/m2)
  Type fast_prev = fast_dat(0); // Initial fast coral cover (%)
  Type slow_prev = slow_dat(0); // Initial slow coral cover (%)

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- SMALL CONSTANT FOR NUMERICAL STABILITY ---
  Type eps = Type(1e-8);

  // --- INITIALIZE PREDICTIONS ---
  cots_pred(0) = cots_prev;
  fast_pred(0) = fast_prev;
  slow_pred(0) = slow_prev;

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++) {
    // 1. COTS recruitment: logistic growth, modulated by SST and larval immigration
    Type env_mod = 1 + beta_sst * (sst_dat(t-1) - Type(27.0)); // SST effect (centered at 27C)
    Type immig = imm_eff * cotsimm_dat(t-1); // Immigration effect

    // Resource limitation: carrying capacity depends on coral cover (saturating, Michaelis-Menten)
    Type coral_sum = fast_prev + slow_prev + eps;
    Type K_cots_eff = K_cots * (coral_sum / (K_coral_half + coral_sum + eps)); // COTS K saturates with total coral

    // Outbreak threshold: sharper sigmoid on COTS recruitment (triggers outbreak when coral is high)
    Type outbreak_mod = 1/(1 + exp(-outbreak_sharpness*(coral_sum - outbreak_thresh))); // Outbreak more likely if coral > outbreak_thresh%

    // COTS predation on corals (Type II functional response)
    Type pred_fast = alpha_fast * cots_prev * fast_prev / (fast_prev + Type(5.0) + eps); // Fast coral eaten
    Type pred_slow = alpha_slow * cots_prev * slow_prev / (slow_prev + Type(10.0) + eps); // Slow coral eaten

    // COTS population update
    Type cots_growth = r_cots * cots_prev * (1 - cots_prev/(K_cots_eff+eps)) * env_mod * outbreak_mod;
    Type cots_next = cots_prev + cots_growth + immig - pred_fast*0.05 - pred_slow*0.02; // Small mortality from feeding inefficiency

    // Bound COTS to positive values
    cots_next = CppAD::CondExpGt(cots_next, eps, cots_next, eps);

    // Fast coral update: logistic regrowth minus COTS predation
    Type fast_growth = r_fast * fast_prev * (1 - fast_prev/(K_fast+eps));
    Type fast_next = fast_prev + fast_growth - pred_fast;

    fast_next = CppAD::CondExpGt(fast_next, eps, fast_next, eps);

    // Slow coral update: logistic regrowth minus COTS predation
    Type slow_growth = r_slow * slow_prev * (1 - slow_prev/(K_slow+eps));
    Type slow_next = slow_prev + slow_growth - pred_slow;

    slow_next = CppAD::CondExpGt(slow_next, eps, slow_next, eps);

    // Save predictions
    cots_pred(t) = cots_next;
    fast_pred(t) = fast_next;
    slow_pred(t) = slow_next;

    // Advance state
    cots_prev = cots_next;
    fast_prev = fast_next;
    slow_prev = slow_next;
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Lognormal likelihood, fixed minimum SD for stability
    Type sd_cots = sqrt(sigma_cots*sigma_cots + eps);
    Type sd_fast = sqrt(sigma_fast*sigma_fast + eps);
    Type sd_slow = sqrt(sigma_slow*sigma_slow + eps);

    nll -= dnorm(log(cots_dat(t)+eps), log(cots_pred(t)+eps), sd_cots, true);
    nll -= dnorm(log(fast_dat(t)+eps), log(fast_pred(t)+eps), sd_fast, true);
    nll -= dnorm(log(slow_dat(t)+eps), log(slow_pred(t)+eps), sd_slow, true);
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: penalize negative growth rates, unreasonably high K, etc.
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots-Type(0.01), Type(0)), 2) * 10.0;
  nll += pow(CppAD::CondExpGt(K_cots, Type(10.0), K_cots-Type(10.0), Type(0)), 2) * 10.0;
  nll += pow(CppAD::CondExpLt(r_fast, Type(0.01), r_fast-Type(0.01), Type(0)), 2) * 10.0;
  nll += pow(CppAD::CondExpLt(r_slow, Type(0.01), r_slow-Type(0.01), Type(0)), 2) * 10.0;

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}
