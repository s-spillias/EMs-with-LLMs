#include <TMB.hpp>

// Crown-of-Thorns Starfish (COTS) Outbreak Model
// Booms and busts with feedback on coral communities
// See intention.txt for ecological description

template<class Type>
Type objective_function<Type>::operator() ()
{
  // === DATA ===
  DATA_VECTOR(Year);     // Time vector (years)
  DATA_VECTOR(cots_dat); // Observed adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%)

  // === PARAMETERS ===
  PARAMETER(r_cots);     // Intrinsic growth rate of adult COTS (per year)
  PARAMETER(K_cots);     // Carrying capacity of COTS (ind/m2)
  PARAMETER(alpha_fast); // Predation rate of COTS on fast corals (per % cover)
  PARAMETER(alpha_slow); // Predation rate of COTS on slow corals (per % cover)
  PARAMETER(g_fast);     // Growth rate of fast-growing coral (% per year)
  PARAMETER(g_slow);     // Growth rate of slow-growing coral (% per year)
  PARAMETER(sigma_proc); // Process error SD
  PARAMETER(sigma_obs);  // Observation error SD

  int n = cots_dat.size();

  // === STATES ===
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // === INITIAL CONDITIONS ===
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  Type eps = Type(1e-8);

  // === PROCESS MODEL ===
  for(int t=1; t<n; t++) {

    // (1) COTS population dynamics: logistic growth with density-dependence
    Type growth_term = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/(K_cots+eps));

    // (2) Coral-dependent survival: reduced growth when food availability declines
    Type coral_food = fast_pred(t-1) + 0.5 * slow_pred(t-1); // weighted availability
    Type food_mod = coral_food / (coral_food + 10.0); // saturating functional response

    // Updated starfish population
    cots_pred(t) = cots_pred(t-1) + growth_term * food_mod;

    // (3) Coral dynamics: logistic growth minus COTS predation
    Type fast_growth = g_fast * fast_pred(t-1) * (1 - (fast_pred(t-1)+slow_pred(t-1))/100.0);
    Type fast_loss = alpha_fast * cots_pred(t-1) * fast_pred(t-1);

    Type slow_growth = g_slow * slow_pred(t-1) * (1 - (fast_pred(t-1)+slow_pred(t-1))/100.0);
    Type slow_loss = alpha_slow * cots_pred(t-1) * slow_pred(t-1);

    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_loss;
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_loss;

    // Bound corals to prevent negative cover
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), eps, fast_pred(t), eps);
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), eps, slow_pred(t), eps);
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), eps, cots_pred(t), eps);

  }

  // === LIKELIHOOD ===
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive observations
    nll -= dnorm(log(cots_dat(t)+eps), log(cots_pred(t)+eps), sigma_obs, true);
    nll -= dnorm(log(fast_dat(t)+eps), log(fast_pred(t)+eps), sigma_obs, true);
    nll -= dnorm(log(slow_dat(t)+eps), log(slow_pred(t)+eps), sigma_obs, true);
  }

  // === REPORTING ===
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
