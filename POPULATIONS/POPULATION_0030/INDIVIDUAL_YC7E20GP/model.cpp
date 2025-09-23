#include <TMB.hpp>

// This TMB model captures outbreak dynamics of Crown-of-Thorns Starfish (COTS) and their effects on coral communities.
// Equations are designed to capture the boom-bust cycles of COTS populations and selective coral predation.

template<class Type>
Type objective_function<Type>::operator() () {

  // ===== DATA INPUTS =====
  DATA_VECTOR(Year);              // Year time series
  DATA_VECTOR(cots_dat);          // Observed COTS abundance (indiv/m^2)
  DATA_VECTOR(fast_dat);          // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat);          // Observed slow coral cover (%)
  DATA_VECTOR(sst_dat);           // Observed SST (deg C)
  DATA_VECTOR(cotsimm_dat);       // Observed larval immigration input (indiv/m^2/yr)

  int n = Year.size();

  // ===== PARAMETERS =====
  PARAMETER(r_cots);              // Intrinsic COTS growth rate (1/yr)
  PARAMETER(K_cots);              // COTS carrying capacity (indiv/m^2)
  PARAMETER(alpha_fast);          // COTS consumption rate on fast coral
  PARAMETER(alpha_slow);          // COTS consumption rate on slow coral
  PARAMETER(r_fast);              // Growth rate of fast corals
  PARAMETER(r_slow);              // Growth rate of slow corals
  PARAMETER(sst_effect);          // Effect of SST anomalies on COTS growth
  PARAMETER(immigration_scale);   // Scaling factor for larval immigration
  PARAMETER(obs_sd);              // Observation error SD (lognormal)
  PARAMETER(food_threshold);      // Minimum survival fraction when coral cover is low

  // Add constants for stability
  Type eps = Type(1e-8);

  // ===== STATE VARIABLES (PREDICTIONS) =====
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // ===== INITIAL CONDITIONS (match first observation) =====
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ===== PROCESS MODEL =====
  for(int t=1; t<n; t++){

    // COTS Population Growth (logistic with SST effect and immigration)
    Type growth_rate = r_cots + sst_effect * (sst_dat(t-1) - 27.0);  // relative to baseline SST=27
    Type cots_growth = cots_pred(t-1) * exp(growth_rate * (1 - cots_pred(t-1)/ (K_cots + eps)));
    Type immigration = immigration_scale * cotsimm_dat(t-1);

    // Coral availability factor (limits COTS survival)
    Type coral_avail = (fast_pred(t-1) + slow_pred(t-1)) / (100.0 + eps);
    Type food_factor = coral_avail + food_threshold;

    // Coral Predation impact
    Type pred_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1)/(1 + fast_pred(t-1)); // saturating
    Type pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1)/(1 + slow_pred(t-1));

    // Coral dynamics (logistic recovery with predation loss)
    Type fast_growth = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1)+slow_pred(t-1))/100.0);
    Type slow_growth = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - (fast_pred(t-1)+slow_pred(t-1))/100.0);

    fast_pred(t) = CppAD::CondExpGt(fast_growth - pred_fast, eps, fast_growth - pred_fast, eps);
    slow_pred(t) = CppAD::CondExpGt(slow_growth - pred_slow, eps, slow_growth - pred_slow, eps);

    // Update COTS with coral-dependent survival
    cots_pred(t) = CppAD::CondExpGt((cots_growth + immigration) * food_factor,
                                    eps,
                                    (cots_growth + immigration) * food_factor,
                                    eps);
  }

  // ===== LIKELIHOOD =====
  Type nll = 0.0;
  Type minsd = 0.05;  // minimum standard deviation

  for(int t=0; t<n; t++){
    nll -= dnorm(log(cots_dat(t) + eps),
                 log(cots_pred(t) + eps),
                 obs_sd + minsd,
                 true);

    nll -= dnorm(log(fast_dat(t) + eps),
                 log(fast_pred(t) + eps),
                 obs_sd + minsd,
                 true);

    nll -= dnorm(log(slow_dat(t) + eps),
                 log(slow_pred(t) + eps),
                 obs_sd + minsd,
                 true);
  }

  // ===== REPORTING =====
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
