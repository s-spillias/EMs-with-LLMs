#include <TMB.hpp>
#include <cppad/cppad.hpp>

template <class Type>
Type objective_function<Type>::operator()()
{
  // ------------------------------------------------------------------------
  // DATA INPUT
  // ------------------------------------------------------------------------
  // ------------------------------------------------------------------------

  DATA_VECTOR(Year);              // Time variable (year)
  DATA_VECTOR(cots_dat);          // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);          // Slow-growing coral cover data (% cover)
  DATA_VECTOR(fast_dat);          // Fast-growing coral cover data (% cover)
  DATA_VECTOR(sst_dat);           // Sea-Surface Temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);       // COTS larval immigration rate (individuals/m2/year)

  // ------------------------------------------------------------------------
  // PARAMETER INPUT
  // ------------------------------------------------------------------------

  PARAMETER(r_max);             // Maximum recruitment rate of COTS (year^-1)
  PARAMETER(K_cots);            // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(d_cots);            // Natural mortality rate of COTS (year^-1)
  PARAMETER(e_cots);            // Consumption rate of coral by COTS (m^2/COTS/year)
  PARAMETER(r_slow);            // Growth rate of slow-growing corals (year^-1)
  PARAMETER(r_fast);            // Growth rate of fast-growing corals (year^-1)
  PARAMETER(sigma_cots);        // Standard deviation of COTS abundance observations
  PARAMETER(sigma_slow);        // Standard deviation of slow-growing coral cover observations
  PARAMETER(sigma_fast);        // Standard deviation of fast-growing coral cover observations

  // ------------------------------------------------------------------------
  // MODEL PARAMETERS
  // ------------------------------------------------------------------------

  int n = Year.size();            // Number of time steps

  // ------------------------------------------------------------------------
  // INITIALIZE STATE VARIABLES
  // ------------------------------------------------------------------------

  vector<Type> cots(n);          // COTS abundance (individuals/m^2)
  vector<Type> slow(n);          // Slow-growing coral cover (% cover)
  vector<Type> fast(n);          // Fast-growing coral cover (% cover)

  // ------------------------------------------------------------------------
  // INITIALIZE PREDICTED VARIABLES
  // ------------------------------------------------------------------------

  vector<Type> cots_pred(n);     // Predicted COTS abundance (individuals/m^2)
  vector<Type> slow_pred(n);     // Predicted slow-growing coral cover (% cover)
  vector<Type> fast_pred(n);     // Predicted fast-growing coral cover (% cover)

  // ------------------------------------------------------------------------
  // LIKELIHOOD
  // ------------------------------------------------------------------------

  Type nll = 0.0;                 // Negative log-likelihood

  // ------------------------------------------------------------------------
  // EQUATIONS
  // ------------------------------------------------------------------------

  // 1. Initial conditions
  cots(0) = cots_dat(0);
  slow(0) = slow_dat(0);
  fast(0) = fast_dat(0);

  // ------------------------------------------------------------------------
  // DYNAMIC EQUATIONS
  // ------------------------------------------------------------------------

  for (int t = 1; t < n; t++) {

    // 3. COTS recruitment
    Type recruitment = r_max / (1 + cots(t-1) / K_cots);

    // 4. Total coral cover
    Type total_coral = slow(t-1) + fast(t-1) + Type(1e-8);

    // 5. COTS mortality
    Type mortality = d_cots * cots(t-1);

    // 6. COTS consumption of slow-growing coral
    Type consumption_slow = e_cots * cots(t-1) * (slow(t-1) / total_coral);

    // 7. COTS consumption of fast-growing coral
    Type consumption_fast = e_cots * cots(t-1) * (fast(t-1) / total_coral);

    // 8. COTS dynamics
    cots(t) = cots(t-1) + recruitment - mortality;
    cots_pred(t) = cots(t);

    // 9. Slow-growing coral dynamics
    slow(t) = slow(t-1) + r_slow * slow(t-1) - consumption_slow;
    slow_pred(t) = slow(t);

    // 10. Fast-growing coral dynamics
    fast(t) = fast(t-1) + r_fast * fast(t-1) - consumption_fast;
    fast_pred(t) = fast(t);

    // ------------------------------------------------------------------------
    // LIKELIHOOD CONTRIBUTION
    // ------------------------------------------------------------------------

    // 11. Likelihood contribution from COTS data
    nll -= dnorm(cots_dat(t), cots_pred(t), sigma_cots, true);

    // 12. Likelihood contribution from slow-growing coral data
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow, true);

    // 13. Likelihood contribution from fast-growing coral data
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast, true);

    // Parameter bounds using smooth penalties
    Type r_max_penalty = 0.0;
    if (r_max < 0.0) r_max_penalty = pow(r_max, 2);
    if (r_max > 10.0) r_max_penalty = pow(r_max - 10.0, 2);
    nll += r_max_penalty;

    Type d_cots_penalty = 0.0;
    if (d_cots < 0.0) d_cots_penalty = pow(d_cots, 2);
    if (d_cots > 1.0) d_cots_penalty = pow(d_cots - 1.0, 2);
    nll += d_cots_penalty;

    Type e_cots_penalty = 0.0;
    if (e_cots < 0.0) e_cots_penalty = pow(e_cots, 2);
    if (e_cots > 1.0) e_cots_penalty = pow(e_cots - 1.0, 2);
    nll += e_cots_penalty;

      Type r_slow_penalty = 0.0;
    if (r_slow < -1.0) r_slow_penalty = pow(r_slow + 1.0, 2);
    if (r_slow > 10.0) r_slow_penalty = pow(r_slow - 10.0, 2);
    nll += r_slow_penalty;

    Type r_fast_penalty = 0.0;
    if (r_fast < -1.0) r_fast_penalty = pow(r_fast + 1.0, 2);
    if (r_fast > 10.0) r_fast_penalty = pow(r_fast - 10.0, 2);
    nll += r_fast_penalty;
  }

  // ------------------------------------------------------------------------
  // REPORT SECTION
  // ------------------------------------------------------------------------

  ADREPORT(cots_pred);
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  REPORT(cots);
  REPORT(slow);
  REPORT(fast);

  return nll;
}
