#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // 1. Data and Parameters:
  // ------------------------------------------------------------------------

  // --- Data: ---
  DATA_VECTOR(Year);              // Time variable (year)
  DATA_VECTOR(cots_dat);          // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);          // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);          // Fast-growing coral cover data (%)
  DATA_VECTOR(sst_dat);           // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);       // COTS larval immigration rate (individuals/m2/year)

  // --- Parameters: ---
  PARAMETER(log_r_cots);          // Log of intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots);          // Log of carrying capacity of COTS (individuals/m2)
  PARAMETER(log_m_cots);          // Log of natural mortality rate of COTS (year^-1)
  PARAMETER(log_p_cots);          // Log of predation rate on COTS (year^-1)
  PARAMETER(log_K1_cots);         // Log of half-saturation constant for COTS predation (individuals/m2)
  PARAMETER(log_a_fast);         // Log of attack rate of COTS on fast-growing coral (m2/individual/year)
  PARAMETER(log_a_slow);         // Log of attack rate of COTS on slow-growing coral (m2/individual/year)
  PARAMETER(log_K_fast);         // Log of carrying capacity of fast-growing coral (%)
  PARAMETER(log_K_slow);         // Log of carrying capacity of slow-growing coral (%)
  PARAMETER(log_r_fast);         // Log of growth rate of fast-growing coral (year^-1)
  PARAMETER(log_r_slow);         // Log of growth rate of slow-growing coral (year^-1)
  PARAMETER(log_m_fast);         // Log of mortality rate of fast-growing coral (year^-1)
  PARAMETER(log_m_slow);         // Log of mortality rate of slow-growing coral (year^-1)
  PARAMETER(log_temp_sensitivity_fast); // Log of temperature sensitivity of fast-growing coral (Celsius^-1)
  PARAMETER(log_temp_sensitivity_slow); // Log of temperature sensitivity of slow-growing coral (Celsius^-1)
  PARAMETER(log_sigma_cots);      // Log of standard deviation of COTS observation error
  PARAMETER(log_sigma_slow);      // Log of standard deviation of slow-growing coral observation error
  PARAMETER(log_sigma_fast);      // Log of standard deviation of fast-growing coral observation error
  PARAMETER(log_h_cots);           // Log of handling time for COTS predation (year)
  PARAMETER(log_K2_cots);          // Log of COTS density at which interference effects become significant (individuals/m2)

  // --- Transformations: ---
  Type r_cots   = exp(log_r_cots);
  Type K_cots   = exp(log_K_cots);
  Type m_cots   = exp(log_m_cots);
  Type p_cots   = exp(log_p_cots);
  Type K1_cots  = exp(log_K1_cots);
  Type a_fast  = exp(log_a_fast);
  Type a_slow  = exp(log_a_slow);
  Type K_fast   = exp(log_K_fast);
  Type K_slow   = exp(log_K_slow);
  Type r_fast   = exp(log_r_fast);
  Type r_slow   = exp(log_r_slow);
  Type m_fast   = exp(log_m_fast);
  Type m_slow   = exp(log_m_slow);
  Type temp_sensitivity_fast = exp(log_temp_sensitivity_fast);
  Type temp_sensitivity_slow = exp(log_temp_sensitivity_slow);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  Type h_cots    = exp(log_h_cots);
  Type K2_cots   = exp(log_K2_cots);

  // --- Objective function: ---
  Type nll = 0.0; // Initialize negative log-likelihood

  // --- Vectors for predictions: ---
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());

  // --- Initial conditions: ---
  cots_pred(0) = cots_dat(0);    // Initial COTS abundance
  slow_pred(0) = slow_dat(0);    // Initial slow-growing coral cover
  fast_pred(0) = fast_dat(0);    // Initial fast-growing coral cover

  // ------------------------------------------------------------------------
  // 2. Model Equations:
  // ------------------------------------------------------------------------

  for(int t=1; t<Year.size(); t++) {
    // 1. COTS Population Dynamics:
    //    Logistic growth with carrying capacity, predation, and larval immigration.
    Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / K_cots);
    Type cots_predation = p_cots * cots_pred(t-1) / (K1_cots + cots_pred(t-1) + (pow(cots_pred(t-1), 2.0) / K2_cots)); // Density-dependent Holling type II
    cots_pred(t) = cots_pred(t-1) + cots_growth - m_cots * cots_pred(t-1) - cots_predation + cotsimm_dat(t);

    // 2. Coral Dynamics:
    //    Logistic growth with COTS predation and temperature-dependent mortality.
    // COTS preferentially eat fast-growing coral.
    // Holling type III functional response:
    Type fast_predation = a_fast * pow(cots_pred(t-1), 2.0) * fast_pred(t-1) / (pow(h_cots, 2.0) + pow(cots_pred(t-1), 2.0));
    Type temp_mortality_fast = temp_sensitivity_fast * sst_dat(t) * fast_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1) / (K_fast + slow_pred(t-1))) * (fast_pred(t-1) > Type(0.0)) - m_fast * fast_pred(t-1) - fast_predation - temp_mortality_fast;

    Type slow_predation = a_slow * pow(cots_pred(t-1), 2.0) * slow_pred(t-1) / (pow(h_cots, 2.0) + pow(cots_pred(t-1), 2.0));
    Type temp_mortality_slow = temp_sensitivity_slow * sst_dat(t) * slow_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1) / (K_slow + fast_pred(t-1))) * (slow_pred(t-1) > Type(0.0)) - m_slow * slow_pred(t-1) - slow_predation - temp_mortality_slow;

    // --- Add small constant to prevent negative values ---
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
  }

  // ------------------------------------------------------------------------
  // 3. Likelihood Calculation:
  // ------------------------------------------------------------------------

  for(int t=0; t<Year.size(); t++) {
    // --- COTS likelihood: ---
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);

    // --- Slow-growing coral likelihood: ---
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_slow, true);

    // --- Fast-growing coral likelihood: ---
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_fast, true);

    // --- Parameter penalties: ---
    nll += Type(0.01) * pow(log_r_cots - 0.5, 2.0);
    nll += Type(0.01) * pow(log_K_cots - 2.0, 2.0);
    nll += Type(0.01) * pow(log_m_cots + 0.5, 2.0);
    nll += Type(0.01) * pow(log_p_cots + 1.0, 2.0);
    nll += Type(0.01) * pow(log_K1_cots + 0.5, 2.0);
    nll += Type(0.01) * pow(log_a_fast + 2.0, 2.0);
    nll += Type(0.01) * pow(log_a_slow + 3.0, 2.0);
    nll += Type(0.01) * pow(log_K_fast - 3.0, 2.0);
    nll += Type(0.01) * pow(log_K_slow - 3.0, 2.0);
    nll += Type(0.01) * pow(log_r_fast - 0.8, 2.0);
    nll += Type(0.01) * pow(log_r_slow - 0.3, 2.0);
    nll += Type(0.01) * pow(log_m_fast + 0.2, 2.0);
    nll += Type(0.01) * pow(log_m_slow + 0.7, 2.0);
    nll += Type(0.01) * pow(log_temp_sensitivity_fast + 3.0, 2.0);
    nll += Type(0.01) * pow(log_temp_sensitivity_slow + 3.0, 2.0);
    nll += Type(0.01) * pow(log_sigma_cots + 0.5, 2.0);
    nll += Type(0.01) * pow(log_sigma_slow + 0.5, 2.0);
    nll += Type(0.01) * pow(log_sigma_fast + 0.5, 2.0);
  }

  // ------------------------------------------------------------------------
  // 4. Reporting:
  // ------------------------------------------------------------------------

  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
