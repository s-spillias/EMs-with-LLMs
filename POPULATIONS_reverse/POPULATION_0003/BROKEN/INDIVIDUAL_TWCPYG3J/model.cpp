#include <TMB.hpp>
// A simple model for Crown-of-Thorns starfish dynamics using external forcing.
// Data:
//   sst_dat: Sea-Surface Temperature (°C)
//   cotsimm_dat: Crown-of-thorns immigration rate (ind./m^2/year)
//   cots_dat: Observed crown-of-thorns starfish density (ind./m^2)
// Parameters:
//   intercept: Baseline log-density for starfish (log-scale)
//   beta_sst: Effect of sea-surface temperature on starfish log-density
//   beta_imm: Effect of immigration rate on starfish log-density
//   log_sigma: Log-standard deviation for observation error
template<class Type>
Type objective_function<Type>::operator() () {
  DATA_VECTOR(sst_dat);
  DATA_VECTOR(cotsimm_dat);
  DATA_VECTOR(cots_dat);
  int n = sst_dat.size();
  
  PARAMETER(intercept);
  PARAMETER(beta_sst);
  PARAMETER(beta_imm);
  PARAMETER(log_sigma);
  Type sigma = exp(log_sigma) + Type(1e-8);
  
  Type nll = 0.0;
  vector<Type> cots_pred(n);
  
  // Dynamic recursive prediction for Crown-of-Thorns starfish using only external forcing.
  cots_pred(0) = exp(intercept + beta_sst * sst_dat(0) + beta_imm * cotsimm_dat(0));
  nll -= dnorm(log(cots_dat(0) + Type(1e-8)), log(cots_pred(0) + Type(1e-8)), sigma, true);
  for(int i = 1; i < n; i++){
    cots_pred(i) = cots_pred(i-1) * exp(beta_sst * sst_dat(i) + beta_imm * cotsimm_dat(i));
    nll -= dnorm(log(cots_dat(i) + Type(1e-8)), log(cots_pred(i) + Type(1e-8)), sigma, true);
  }
  
  ADREPORT(cots_pred);
  ADREPORT(intercept);
  ADREPORT(beta_sst);
  ADREPORT(beta_imm);
  ADREPORT(sigma);
  
  return nll;
}
