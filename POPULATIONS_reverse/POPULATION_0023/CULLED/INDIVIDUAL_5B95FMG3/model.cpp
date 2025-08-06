#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // 1. DATA INPUT
  // ------------------------------------------------------------------------

  DATA_VECTOR(Year);              // Years of simulation (years)
  DATA_VECTOR(sst_dat);           // Sea Surface Temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);       // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);          // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);          // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);          // Fast-growing coral cover data (%)

  // ------------------------------------------------------------------------
  // 2. PARAMETER INPUT
  // ------------------------------------------------------------------------

  PARAMETER(log_r_cots);          // Log of intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots);          // Log of carrying capacity of COTS (individuals/m2)
  PARAMETER(log_alpha_cots_slow); // Log of COTS predation rate on slow-growing coral (% cover reduction per COTS per year)
  PARAMETER(log_alpha_cots_fast); // Log of COTS predation rate on fast-growing coral (% cover reduction per COTS per year)
  PARAMETER(log_r_slow);          // Log of intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(log_K_slow);          // Log of carrying capacity of slow-growing coral (% cover)
  PARAMETER(log_r_fast);          // Log of intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(log_K_fast);          // Log of carrying capacity of fast-growing coral (% cover)
  PARAMETER(log_sigma_cots);      // Log of standard deviation of COTS observation error
  PARAMETER(log_sigma_slow);      // Log of standard deviation of slow-growing coral observation error
  PARAMETER(log_sigma_fast);      // Log of standard deviation of fast-growing coral observation error
  PARAMETER(temp_effect);         // Effect of temperature on COTS growth (increase in growth rate per degree Celsius)
  PARAMETER(outbreak_trigger);    // COTS density required to trigger an outbreak (individuals/m2)
  PARAMETER(log_mortality_cots);  // Log of density-dependent mortality rate of COTS (individuals/m2/year)
  PARAMETER(recovery_rate);       // Rate at which coral recovers after COTS decline (year^-1)

  // ------------------------------------------------------------------------
  // 3. TRANSFORM PARAMETERS
  // ------------------------------------------------------------------------

  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_cots_slow = exp(log_alpha_cots_slow);
  Type alpha_cots_fast = exp(log_alpha_cots_fast);
  Type r_slow = exp(log_r_slow);
  Type K_slow = exp(log_K_slow);
  Type r_fast = exp(log_r_fast);
  Type K_fast = exp(log_K_fast);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  Type mortality_cots = exp(log_mortality_cots);

  // ------------------------------------------------------------------------
  // 4. SET UP OBJECTIVE FUNCTION
  // ------------------------------------------------------------------------

  Type nll = 0.0; // Initialize negative log-likelihood

  // ------------------------------------------------------------------------
  // 5. DEFINE STATE VARIABLES
  // ------------------------------------------------------------------------

  int n = Year.size();
  vector<Type> cots(n);          // COTS abundance (individuals/m2)
  vector<Type> slow(n);          // Slow-growing coral cover (%)
  vector<Type> fast(n);          // Fast-growing coral cover (%)
  vector<Type> cots_pred(n);     // Predicted COTS abundance (individuals/m2)
  vector<Type> slow_pred(n);     // Predicted slow-growing coral cover (%)
  vector<Type> fast_pred(n);     // Predicted fast-growing coral cover (%)

  // ------------------------------------------------------------------------
  // 6. INITIALIZE STATE VARIABLES
  // ------------------------------------------------------------------------

  cots[0] = cots_dat[0];       // Initial COTS abundance from data
  slow[0] = slow_dat[0];       // Initial slow-growing coral cover from data
  fast[0] = fast_dat[0];       // Initial fast-growing coral cover from data

  // ------------------------------------------------------------------------
  // 7. POPULATION DYNAMICS MODEL
  // ------------------------------------------------------------------------

  for(int t = 1; t < n; t++) {

    // 1. COTS Population Dynamics
    Type temp_adjust = exp(temp_effect * (sst_dat[t-1] - 27.0)); // Temperature effect, baseline 27C
    Type cots_growth = r_cots * temp_adjust * cots[t-1] * (1 - cots[t-1] / K_cots); // Logistic growth with temp effect
    if (cots[t-1] > outbreak_trigger) {
      cots_growth *= 5;  // Boost growth during outbreak
    }
    Type cots_mortality = mortality_cots * cots[t-1] * cots[t-1]; // Density-dependent mortality
    cots[t] = cots[t-1] + cots_growth - alpha_cots_slow * cots[t-1] * slow[t-1] - alpha_cots_fast * cots[t-1] * fast[t-1] - cots_mortality + cotsimm_dat[t-1]; // COTS dynamics with predation, immigration, and density-dependent mortality
    cots[t] = (cots[t] > Type(0.0)) ? cots[t] : Type(0.0); // Prevent negative abundance

    // 2. Slow-Growing Coral Dynamics
    slow[t] = slow[t-1] + r_slow * slow[t-1] * (1 - slow[t-1] / K_slow) - alpha_cots_slow * cots[t-1] * slow[t-1] + recovery_rate * (K_slow - slow[t-1]); // Coral growth, COTS predation, and recovery
    slow[t] = (slow[t] > Type(0.0)) ? slow[t] : Type(0.0); // Prevent negative cover
    slow[t] = (slow[t] < Type(100.0)) ? slow[t] : Type(100.0); // Prevent cover exceeding 100%

    // 3. Fast-Growing Coral Dynamics
    fast[t] = fast[t-1] + r_fast * fast[t-1] * (1 - fast[t-1] / K_fast) - alpha_cots_fast * cots[t-1] * fast[t-1] + recovery_rate * (K_fast - fast[t-1]); // Coral growth, COTS predation, and recovery
    fast[t] = (fast[t] > Type(0.0)) ? fast[t] : Type(0.0); // Prevent negative cover
    fast[t] = (fast[t] < Type(100.0)) ? fast[t] : Type(100.0); // Prevent cover exceeding 100%

    // 4. PREDICTIONS
    cots_pred[t] = cots[t];
    slow_pred[t] = slow[t];
    fast_pred[t] = fast[t];
  }

  // ------------------------------------------------------------------------
  // 8. LIKELIHOOD CONTRIBUTION
  // ------------------------------------------------------------------------

  Type min_sigma = Type(0.01); // Minimum standard deviation

  for(int t = 0; t < n; t++) {
    // COTS Likelihood
    Type sigma_cots_t = (sigma_cots > min_sigma) ? sigma_cots : min_sigma;
    nll -= dnorm(log(cots_dat[t] + Type(1e-8)), log(cots_pred[t] + Type(1e-8)), sigma_cots_t, true);

    // Slow-Growing Coral Likelihood
    Type sigma_slow_t = (sigma_slow > min_sigma) ? sigma_slow : min_sigma;
    nll -= dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), sigma_slow_t, true);

    // Fast-Growing Coral Likelihood
    Type sigma_fast_t = (sigma_fast > min_sigma) ? sigma_fast : min_sigma;
    nll -= dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), sigma_fast_t, true);
  }

  // ------------------------------------------------------------------------
  // 9. REPORT SECTION
  // ------------------------------------------------------------------------

  REPORT(cots);
  REPORT(slow);
  REPORT(fast);
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  // ------------------------------------------------------------------------
  // 10. RETURN OBJECTIVE FUNCTION VALUE
  // ------------------------------------------------------------------------

  return nll;
}
