#include <TMB.hpp>
using namespace density;
using namespace Eigen;

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                // Years of data (time variable)
  DATA_VECTOR(cots_dat);            // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);            // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);            // Fast-growing coral cover data (%)
  DATA_VECTOR(sst_dat);             // Sea Surface Temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);         // COTS larval immigration rate (individuals/m2/year)

  // PARAMETERS
  PARAMETER(log_cots_r);          // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_cots_K);          // Carrying capacity of COTS (individuals/m2)
  PARAMETER(log_cots_sigma);      // Observation error for COTS abundance
  PARAMETER(log_slow_r);          // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(log_slow_K);          // Carrying capacity of slow-growing coral (%)
  PARAMETER(log_slow_sigma);      // Observation error for slow-growing coral cover
  PARAMETER(log_fast_r);          // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(log_fast_K);          // Carrying capacity of fast-growing coral (%)
  PARAMETER(log_fast_sigma);      // Observation error for fast-growing coral cover
  PARAMETER(cots_eat_slow);       // Predation rate of COTS on slow-growing coral (proportion eaten per COTS per year)
  PARAMETER(cots_eat_fast);       // Predation rate of COTS on fast-growing coral (proportion eaten per COTS per year)
  PARAMETER(sst_effect);          // Effect of SST on COTS larval survival (increase in growth rate per degree Celsius)
  PARAMETER(cotsimm_effect);      // Effect of larval immigration on COTS growth rate (increase in growth rate per individual/m2/year)
  PARAMETER(log_cots_mortality);  // Natural mortality rate of COTS (year^-1)
  PARAMETER(log_power);           // Power parameter for carrying capacity
  PARAMETER(log_cots_density_mortality); // Coefficient for density-dependent mortality of COTS (year^-1)
  PARAMETER(log_hill_K);           // COTS density at which recruitment is half-maximal (individuals/m2)
  PARAMETER(hill_power);            // Steepness of the density dependence in COTS recruitment

  // Transformations
  Type cots_r = exp(log_cots_r);
  Type cots_K = exp(log_cots_K);
  Type cots_mortality = exp(log_cots_mortality);
  Type hill_K = exp(log_hill_K);
  Type cots_density_mortality = exp(log_cots_density_mortality);
  Type power = exp(log_power);
  Type cots_sigma = exp(log_cots_sigma);
  Type slow_r = exp(log_slow_r);
  Type slow_K = exp(log_slow_K);
  Type slow_sigma = exp(log_slow_sigma);
  Type fast_r = exp(log_fast_r);
  Type fast_K = exp(log_fast_K);
  Type fast_sigma = exp(log_fast_sigma);

  // Variables
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);

  // Initialize
  cots_pred(0) = cots_dat(0);    // Initial COTS abundance
  slow_pred(0) = slow_dat(0);    // Initial slow-growing coral cover
  fast_pred(0) = fast_dat(0);    // Initial fast-growing coral cover

  Type nll = 0.0;  // Negative log-likelihood

  // Equations:
  // 1. COTS population dynamics: dN/dt = r * N * (1 - N/K) + sst_effect * SST * N + cotsimm_effect * Immigration * N - predation
  // 2. Slow-growing coral dynamics: dS/dt = r * S * (1 - S/K) - cots_eat_slow * COTS * S
  // 3. Fast-growing coral dynamics: dF/dt = r * F * (1 - F/K) - cots_eat_fast * COTS * F

  for(int i = 1; i < n; i++) {
    // COTS dynamics
    Type sst_adj = sst_effect * sst_dat(i-1);  // SST effect on COTS growth
    Type immigration_adj = cotsimm_effect * cotsimm_dat(i-1); // Immigration effect on COTS growth
    Type density_dependence = pow(cots_pred(i-1), hill_power) / (pow(cots_pred(i-1), hill_power) + pow(hill_K, hill_power)); // Density-dependent recruitment
    Type carrying_capacity = 1.0 / (1.0 + pow(cots_pred(i-1) / cots_K, power)); // Carrying capacity effect
    Type cots_growth = cots_r * cots_pred(i-1) * carrying_capacity + sst_adj * cots_pred(i-1) + immigration_adj * cots_pred(i-1) * density_dependence;
    Type cots_predation = cots_eat_slow * cots_pred(i-1) * slow_pred(i-1) + cots_eat_fast * cots_pred(i-1) * fast_pred(i-1); // Total predation on coral
    Type cots_density_dependent_mortality = cots_density_mortality * cots_pred(i-1) * cots_pred(i-1);
    cots_pred(i) = cots_pred(i-1) + cots_growth - cots_predation - cots_mortality * cots_pred(i-1) - cots_density_dependent_mortality;
    cots_pred(i) = (cots_pred(i) > 0.0) ? cots_pred(i) : Type(1e-8); // Prevent negative abundance

    // Slow-growing coral dynamics
    Type slow_growth = slow_r * slow_pred(i-1) * (1 - slow_pred(i-1) / slow_K);
    Type slow_predation = cots_eat_slow * cots_pred(i-1) * slow_pred(i-1);
    slow_pred(i) = slow_pred(i-1) + slow_growth - slow_predation;
    slow_pred(i) = (slow_pred(i) > 0.0) ? slow_pred(i) : Type(1e-8); // Prevent negative cover

    // Fast-growing coral dynamics
    Type fast_growth = fast_r * fast_pred(i-1) * (1 - fast_pred(i-1) / fast_K);
    Type fast_predation = cots_eat_fast * cots_pred(i-1) * fast_pred(i-1);
    fast_pred(i) = fast_pred(i-1) + fast_growth - fast_predation;
    fast_pred(i) = (fast_pred(i) > 0.0) ? fast_pred(i) : Type(1e-8); // Prevent negative cover

    // Likelihood calculation
    nll -= dnorm(log(cots_dat(i)), log(cots_pred(i)), cots_sigma, true);
    nll -= dnorm(log(slow_dat(i)), log(slow_pred(i)), slow_sigma, true);
    nll -= dnorm(log(fast_dat(i)), log(fast_pred(i)), fast_sigma, true);

    // Penalize predation rates outside [0, 1]
    nll -= dbeta(cots_eat_slow, Type(2.0), Type(2.0), true); // Prior for cots_eat_slow
    nll -= dbeta(cots_eat_fast, Type(2.0), Type(2.0), true); // Prior for cots_eat_fast
  }

  // Reporting
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
