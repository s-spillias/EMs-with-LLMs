#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // 1. Data and Parameters:
  // ------------------------------------------------------------------------

  // Data:
  DATA_VECTOR(Year);                // Years of the simulation
  DATA_VECTOR(sst_dat);             // Sea-Surface Temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);         // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);            // Adult COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);            // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);            // Fast-growing coral cover data (%)

  // Parameters:
  PARAMETER(log_r_cots);            // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots);            // Carrying capacity of COTS (individuals/m2)
  PARAMETER(log_m_cots);            // Natural mortality rate of COTS (year^-1)
  PARAMETER(log_pref_fast);         // Preference of COTS for fast-growing coral
  PARAMETER(log_r_slow);            // Growth rate of slow-growing coral (year^-1)
  PARAMETER(log_r_fast);            // Growth rate of fast-growing coral (year^-1)
  PARAMETER(log_K_slow);            // Carrying capacity of slow-growing coral (%)
  PARAMETER(log_K_fast);            // Carrying capacity of fast-growing coral (%)
  PARAMETER(log_sigma_cots);        // Observation error for COTS abundance
  PARAMETER(log_sigma_slow);        // Observation error for slow-growing coral cover
  PARAMETER(log_sigma_fast);        // Observation error for fast-growing coral cover
  PARAMETER(logit_temp_effect);    // Effect of temperature on COTS recruitment
  PARAMETER(logit_dens_dep);       // Density dependence on COTS recruitment

  // Transformations:
  Type r_cots      = exp(log_r_cots);
  Type K_cots      = exp(log_K_cots);
  Type m_cots      = exp(log_m_cots);
  Type pref_fast   = exp(log_pref_fast);
  Type r_slow      = exp(log_r_slow);
  Type r_fast      = exp(log_r_fast);
  Type K_slow      = exp(log_K_slow);
  Type K_fast      = exp(log_K_fast);
  Type sigma_cots  = exp(log_sigma_cots);
  Type sigma_slow  = exp(log_sigma_slow);
  Type sigma_fast  = exp(log_sigma_fast);
  Type temp_effect = Type(1.0)/(Type(1.0) + exp(-logit_temp_effect)); // between 0 and 1
  Type dens_dep    = Type(1.0)/(Type(1.0) + exp(-logit_dens_dep)); // between 0 and 1

  // ------------------------------------------------------------------------
  // 2. Model Equations:
  // ------------------------------------------------------------------------

  // Define state variables (initial values)
  Type cots(cots_dat(0));       // COTS abundance (individuals/m2)
  Type slow(slow_dat(0));       // Slow-growing coral cover (%)
  Type fast(fast_dat(0));       // Fast-growing coral cover (%)

  // Objective function:
  Type nll = 0.0;

  // Define state variables for the next time step
  Type cots_next = cots;
  Type fast_next = fast;
  Type slow_next = slow;

  // Loop through time:
  for(int t = 1; t < Year.size(); t++) {

    // 1. COTS Dynamics:
    //    Recruitment is influenced by temperature and density dependence
    //    (Equation 1)
    Type recruitment = cotsimm_dat(t) + temp_effect * dens_dep * r_cots * cots * (1 - cots / K_cots);
    Type cots_pred = cots + recruitment - m_cots * cots;

    // 2. Coral Dynamics:
    //    Fast-growing corals are preferred by COTS
    //    (Equation 2)
    Type consumption_fast = pref_fast * cots * fast / (fast + slow + Type(1e-8));
    Type fast_pred = fast + r_fast * fast * (1 - fast / K_fast) - consumption_fast;

    // 3. Slow-growing Coral Dynamics:
    //    (Equation 3)
    Type consumption_slow = cots * slow / (fast + slow + Type(1e-8));
    Type slow_pred = slow + r_slow * slow * (1 - slow / K_slow) - consumption_slow;

    // ------------------------------------------------------------------------
    // 3. Likelihood Calculation:
    // ------------------------------------------------------------------------

    // COTS Likelihood:
    // (Equation 4)
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred + Type(1e-8)), sigma_cots, true);

    // Fast Coral Likelihood:
    // (Equation 5)
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred + Type(1e-8)), sigma_fast, true);

    // Slow Coral Likelihood:
    // (Equation 6)
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred + Type(1e-8)), sigma_slow, true);

    // Update state variables for the next time step
    cots_next = cots_pred > Type(0.0) ? cots_pred : Type(0.0001);
    fast_next = fast_pred > Type(0.0) ? fast_pred : Type(0.0001);
    slow_next = slow_pred > Type(0.0) ? slow_pred : Type(0.0001);

    // Set current state variables to the next time step values
    cots = cots_next;
    fast = fast_next;
    slow = slow_next;
  }

  // ------------------------------------------------------------------------
  // 4. Reporting:
  // ------------------------------------------------------------------------

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
