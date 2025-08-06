#include <TMB.hpp>
template<class Type>
Type dlnorm_pdf(Type x, Type mu, Type sigma) {
  // Returns the log-density of the lognormal distribution:
  // log[f(x)] = -log(x) - log(sigma) - 0.5*log(2*pi) - (log(x)-mu)^2/(2*sigma^2)
  return -log(x) - log(sigma) - 0.5 * log(Type(2.0) * Type(3.141592653589793))
         - pow(log(x) - mu, 2) / (Type(2.0) * sigma * sigma);
}

// 1. Data and Parameters
//    (1) year: Vector of years (data)
//    (2) cots_dat: Observed Crown-of-Thorns starfish abundance (individuals per m2)
//    (3) slow_dat: Observed slow-growing coral cover (percentage)
//    (4) fast_dat: Observed fast-growing coral cover (percentage)
// 
//    (5) log_growth_rate: Log intrinsic coral growth rate (year^-1)
//    (6) log_feeding_rate: Log predation (feeding) rate of COTS on corals (per year per coral unit)
//    (7) log_cots_base: Log baseline COTS abundance (individuals per m2)
// 
// 2. Equations:
//    [1] COTS dynamics: cots_pred[t] = cots_base * exp(-feeding_rate * year[t])
//    [2] Slow-growing corals: slow_pred[t] = slow_dat[t] + growth_rate*(1 - slow_dat[t]/100) - feeding_rate*cots_pred[t]*slow_dat[t]
//    [3] Fast-growing corals: fast_pred[t] = fast_dat[t] + growth_rate*(1 - fast_dat[t]/100) - feeding_rate*cots_pred[t]*fast_dat[t]
//
// Notes: Small constants (Type(1e-8)) are added to avoid division by zero and ensure numerical stability.
//       Lognormal error distributions are used with a fixed minimum standard deviation (0.1) to handle wide-ranging data values.

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;

  // DATA: Vectors of observations
  DATA_VECTOR(cots_dat);           // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);           // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);           // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);            // Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);        // COTS immigration rate (individuals/m2/year)

  // PARAMETERS (log-transformed to ensure positivity)
  PARAMETER(log_feeding_rate);     // Log feeding rate (per year per coral unit); initial value from expert opinion
  PARAMETER(log_cots_base);        // Log baseline COTS abundance (individuals/m2); initial estimate
  
  // Convert log parameters back to original scale
  Type feeding_rate = exp(log_feeding_rate);  // Feeding (predation) rate
  Type cots_base = exp(log_cots_base);          // Baseline COTS abundance (individuals/m2)

  // Likelihood accumulator
  Type nll = 0.0;
  int n = sst_dat.size();

  // Initialize prediction vector for COTS dynamics
  vector<Type> cots_pred(n);     // COTS prediction (individuals/m2)

  // Set initial condition for COTS using baseline
  cots_pred[0] = cots_base;

  // Recursive prediction equation for COTS dynamics using SST and immigration forcing
  for(int t = 1; t < n; t++){
    cots_pred[t] = cots_pred[t-1] * exp(-feeding_rate * sst_dat[t-1]) + cotsimm_dat[t-1];
  }

  // Likelihood calculation for COTS using a lognormal error model (with a fixed minimum SD of 0.1)
  for(int t = 0; t < n; t++){
    nll -= dlnorm_pdf(cots_dat[t], log(cots_pred[t] + Type(1e-8)), Type(0.1));
  }

  // Reporting parameters and predictions
  ADREPORT(feeding_rate);   // Report feeding rate (per year per coral unit)
  ADREPORT(cots_base);      // Report baseline COTS abundance (individuals/m2)
  ADREPORT(cots_pred);      // Report predicted COTS abundances

  return nll;
}
