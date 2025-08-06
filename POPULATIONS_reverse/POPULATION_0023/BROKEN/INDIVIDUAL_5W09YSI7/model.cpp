#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // 1. DATA INPUT
  // ------------------------------------------------------------------------

  DATA_VECTOR(Year);                // Years of the time series
  DATA_VECTOR(cots_dat);            // Observed COTS density (individuals/m^2)
  DATA_VECTOR(slow_dat);            // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);            // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);             // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);         // COTS larval immigration rate (individuals/m^2/year)

  // ------------------------------------------------------------------------
  // 2. PARAMETER INPUT
  // ------------------------------------------------------------------------

  // COTS parameters
  PARAMETER(log_cots_r);            // Log of COTS intrinsic growth rate (year^-1)
  PARAMETER(log_cots_K);            // Log of COTS carrying capacity (individuals/m^2)
  PARAMETER(log_cots_mort);         // Log of COTS natural mortality rate (year^-1)
  PARAMETER(cots_sst_eff);          // Effect of SST on COTS recruitment
  PARAMETER(cots_coral_pref);       // COTS preference for fast-growing coral

  // Coral parameters
  PARAMETER(log_slow_r);            // Log of slow-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_slow_K);            // Log of slow-growing coral carrying capacity (%)
  PARAMETER(log_fast_r);            // Log of fast-growing coral intrinsic growth rate (year^-1)
  PARAMETER(log_fast_K);            // Log of fast-growing coral carrying capacity (%)

  // Observation error parameters
  PARAMETER(log_cots_sd);           // Log of SD for COTS observation error
  PARAMETER(log_slow_sd);           // Log of SD for slow-growing coral observation error
  PARAMETER(log_fast_sd);           // Log of SD for fast-growing coral observation error

  // ------------------------------------------------------------------------
  // 3. TRANSFORM PARAMETERS
  // ------------------------------------------------------------------------

  Type cots_r = exp(log_cots_r);            // COTS intrinsic growth rate (year^-1)
  Type cots_K = exp(log_cots_K);            // COTS carrying capacity (individuals/m^2)
  Type cots_mort = exp(log_cots_mort);         // COTS natural mortality rate (year^-1)
  Type slow_r = exp(log_slow_r);            // slow-growing coral intrinsic growth rate (year^-1)
  Type slow_K = exp(log_slow_K);            // slow-growing coral carrying capacity (%)
  Type fast_r = exp(log_fast_r);            // fast-growing coral intrinsic growth rate (year^-1)
  Type fast_K = exp(log_fast_K);            // fast-growing coral carrying capacity (%)
  Type cots_sd = exp(log_cots_sd);           // SD for COTS observation error
  Type slow_sd = exp(log_slow_sd);           // SD for slow-growing coral observation error
  Type fast_sd = exp(log_fast_sd);           // SD for fast-growing coral observation error

  // ------------------------------------------------------------------------
  // 4. SET UP OBJECTIVE FUNCTION
  // ------------------------------------------------------------------------

  Type nll = 0.0;                           // Initialize negative log-likelihood

  // ------------------------------------------------------------------------
  // 5. DEFINE DYNAMIC EQUATIONS
  // ------------------------------------------------------------------------

  // Initialize state variables
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());

  // Initial conditions (assuming first observation is a reasonable starting point)
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);

  // Add initial condition to NLL
  Type initial_sd = Type(0.1); // Fixed SD for initial conditions
  nll -= dnorm(cots_pred(0), cots_dat(0), cots_sd, true);
  nll -= dnorm(slow_pred(0), slow_dat(0), slow_sd, true);
  nll -= dnorm(fast_pred(0), fast_dat(0), fast_sd, true);

  // 1. COTS dynamics
  // 2. Slow-growing coral dynamics
  // 3. Fast-growing coral dynamics

  for(int t=1; t<Year.size(); t++) {

    // Environmental effect on COTS (e.g., SST impact on recruitment)
    Type sst_effect = exp(cots_sst_eff * sst_dat(t-1));

    // Total coral cover from previous time step
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + Type(1e-8);

    // COTS functional response (Holling type II)
    Type cots_feeding_rate = (cots_r * total_coral) / (cots_K + total_coral + Type(1e-8));

    // 1. COTS dynamics equation
    cots_pred(t) = cots_pred(t-1) + sst_effect * cotsimm_dat(t-1) +
      cots_r * cots_pred(t-1) * (1 - cots_pred(t-1) / cots_K) -
      cots_mort * cots_pred(t-1) - cots_feeding_rate * cots_pred(t-1);
    cots_pred(t) = fmax(Type(0.0), fmin(cots_pred(t), Type(cots_K * 2.0))); // Bound COTS

    // 2. Slow-growing coral dynamics equation
    slow_pred(t) = slow_pred(t-1) + slow_r * slow_pred(t-1) * (1 - (slow_pred(t-1) + fast_pred(t-1)) / slow_K) -
      (1 - cots_coral_pref) * cots_feeding_rate * cots_pred(t-1) * (slow_pred(t-1) / total_coral);
    slow_pred(t) = fmax(Type(0.0), fmin(slow_pred(t), Type(100.0))); // Bound slow coral

    // 3. Fast-growing coral dynamics equation
    fast_pred(t) = fast_pred(t-1) + fast_r * fast_pred(t-1) * (1 - (slow_pred(t-1) + fast_pred(t-1)) / fast_K) -
      cots_coral_pref * cots_feeding_rate * cots_pred(t-1) * (fast_pred(t-1) / total_coral);
    fast_pred(t) = fmax(Type(0.0), fmin(fast_pred(t), Type(100.0))); // Bound fast coral

    // Add observation likelihoods to NLL (using a fixed minimum SD)
    Type min_sd = Type(0.01);  // Minimum SD
    nll -= dnorm(cots_pred(t), cots_dat(t), cots_sd, true);
    nll -= dnorm(log(slow_pred(t)+Type(1e-8)), log(slow_dat(t)+Type(1e-8)), slow_sd, true); // Log-transform slow
    nll -= dnorm(log(fast_pred(t)+Type(1e-8)), log(fast_dat(t)+Type(1e-8)), fast_sd, true); // Log-transform fast
  }

  // ------------------------------------------------------------------------
  // 6. REPORT SECTION
  // ------------------------------------------------------------------------

  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  // ------------------------------------------------------------------------
  // 7. FINAL OUTPUT
  // ------------------------------------------------------------------------

  return nll;
}
