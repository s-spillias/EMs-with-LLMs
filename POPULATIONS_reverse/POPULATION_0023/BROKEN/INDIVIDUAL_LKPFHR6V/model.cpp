#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // 1. Data input
  DATA_VECTOR(Year);             // Time variable (Year)
  DATA_VECTOR(sst_dat);          // Sea Surface Temperature data
  DATA_VECTOR(cotsimm_dat);       // COTS larval immigration data
  DATA_VECTOR(cots_dat);          // COTS abundance data
  DATA_VECTOR(slow_dat);          // Slow-growing coral cover data
  DATA_VECTOR(fast_dat);          // Fast-growing coral cover data

  // 2. Parameters
  PARAMETER(cots_recruitment);   // Recruitment rate of COTS
  PARAMETER(coral_growth_rate);  // Growth rate of coral

  // 3. Objective function
  Type nll = 0.0;                // Negative log-likelihood

  // 4. Model equations
  int n = Year.size();

  // Initialize lagged prediction variables
  Type cots_pred_lag = cots_dat(0); // Initial value from data
  Type slow_pred_lag = slow_dat(0); // Initial value from data
  Type fast_pred_lag = fast_dat(0); // Initial value from data

    // 5. Likelihood contribution: compare current data to *lagged* predictions
    Type sd_cots = Type(0.1); // Fixed standard deviation for COTS
    Type sd_slow = Type(0.1); // Fixed standard deviation for slow coral
    Type sd_fast = Type(0.1); // Fixed standard deviation for fast coral

    nll -= dnorm(cots_dat(0), cots_pred_lag, sd_cots, true); // Likelihood uses initial value
    nll -= dnorm(slow_dat(0), slow_pred_lag, sd_slow, true); // Likelihood uses initial value
    nll -= dnorm(fast_dat(0), fast_pred_lag, sd_fast, true); // Likelihood uses initial value

  for(int i = 1; i < n; i++) {
    // 1. COTS Population Model
    Type cots_pred = cots_dat(i-1) + cots_recruitment * cotsimm_dat(i); // Use previous time step data
    cots_pred = (cots_pred > Type(0.0)) ? cots_pred : Type(0.0); // Ensure positivity

    // 2. Coral Growth Models
    Type slow_pred = slow_dat(i-1) + coral_growth_rate * (Type(1.0) - slow_dat(i-1)/Type(100.0)); // Use previous time step data
    slow_pred = (slow_pred > Type(0.0)) ? slow_pred : Type(0.0);
    slow_pred = (slow_pred < Type(100.0)) ? slow_pred : Type(100.0); // Cap at 100%

    Type fast_pred = fast_dat(i-1) + coral_growth_rate * (Type(1.0) - fast_dat(i-1)/Type(100.0)); // Use previous time step data
    fast_pred = (fast_pred > Type(0.0)) ? fast_pred : Type(0.0);
    fast_pred = (fast_pred < Type(100.0)) ? fast_pred : Type(100.0); // Cap at 100%

    // Update lagged variables for the next iteration
    cots_pred_lag = cots_pred;
    slow_pred_lag = slow_pred;
    fast_pred_lag = fast_pred;

    // 5. Likelihood contribution: compare current data to *lagged* predictions
    nll -= dnorm(cots_dat(i), cots_pred_lag, sd_cots, true); // Likelihood uses lagged prediction
    nll -= dnorm(slow_dat(i), slow_pred_lag, sd_slow, true); // Likelihood uses lagged prediction
    nll -= dnorm(fast_dat(i), fast_pred_lag, sd_fast, true); // Likelihood uses lagged prediction
  }

  // Reporting predicted values
  ADREPORT(cots_pred_lag);
  ADREPORT(slow_pred_lag);
  ADREPORT(fast_pred_lag);

  return nll;
}
