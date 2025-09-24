#include <TMB.hpp>  // TMB header

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(time);                        // 1. Time steps (Year)
  DATA_VECTOR(coralfast_dat);               // 2. Fast-growing coral observations (%)
  DATA_VECTOR(coralslow_dat);               // 3. Slow-growing coral observations (%)
  DATA_VECTOR(cots_dat);                    // 4. Adult COTS density observations (individuals/m2)

  int n = time.size();

  // Parameters (numbered list below)
  // 1. intrinsic_growth_coral: Intrinsic coral growth rate (year^-1)
  // 2. cots_growth_rate: Exponential growth rate of COTS during outbreak phases (year^-1)
  // 3. cots_outbreak_threshold: Threshold COTS density triggering outbreak dynamics (individuals/m2)
  // 4. cots_predation_efficiency: Efficiency of COTS predation on coral (% loss per COTS unit)
  // 5. environment_modifier: Modifier to alter rates based on environmental conditions (unitless)
  // 6. coral_mortality_rate: Natural mortality rate of coral (% loss per year)
  // 7. sd_obs: Observation standard deviation (log-scale error), minimum enforced for numerical stability

  PARAMETER(intrinsic_growth_coral);         // 1.
  PARAMETER(cots_growth_rate);               // 2.
  PARAMETER(cots_outbreak_threshold);        // 3.
  PARAMETER(cots_predation_efficiency);        // 4.
  PARAMETER(environment_modifier);           // 5.
  PARAMETER(coral_mortality_rate);           // 6.
  PARAMETER(log_sd_obs);                     // Log of standard deviation for observations

  Type sd_obs = exp(log_sd_obs) + Type(1e-8);  // Ensure sd_obs is positive, add small constant

  // Initialize predicted values vectors
  vector<Type> coralfast_pred(n);
  vector<Type> coralslow_pred(n);
  vector<Type> cots_pred(n);

  // Set initial conditions using the first observations (avoid data leakage)
  coralfast_pred[0] = coralfast_dat[0];
  coralslow_pred[0] = coralslow_dat[0];
  cots_pred[0] = cots_dat[0];

  // 7. Likelihood objective to be summed
  Type nll = 0.0;

  // Loop from second time-step to predict values using lagged predictions
  for(int t = 1; t < n; t++){
    // 1. Coral fast dynamics with saturating growth and mortality:
    // Equation (1): coralfast_pred = previous + intrinsic growth modified by environment, minus losses due to natural mortality and predation by COTS.
    coralfast_pred[t] = coralfast_pred[t-1]
      + intrinsic_growth_coral * environment_modifier * coralfast_pred[t-1] / (Type(1) + coralfast_pred[t-1])
      - coral_mortality_rate * coralfast_pred[t-1] 
      - cots_predation_efficiency * cots_pred[t-1] * coralfast_pred[t-1] / (coralfast_pred[t-1] + Type(1e-8));  // smooth denominator

    // 2. Coral slow dynamics similarly:
    // Equation (2): coralslow_pred = previous + intrinsic growth modified by environment, minus natural mortality and predation by COTS.
    coralslow_pred[t] = coralslow_pred[t-1]
      + intrinsic_growth_coral * environment_modifier * coralslow_pred[t-1] / (Type(1) + coralslow_pred[t-1])
      - coral_mortality_rate * coralslow_pred[t-1]
      - cots_predation_efficiency * cots_pred[t-1] * coralslow_pred[t-1] / (coralslow_pred[t-1] + Type(1e-8));

    // 3. COTS dynamics with boom-bust behavior:
    // Equation (3): cots_pred = previous + growth if above threshold (using smooth transition), minus density-dependent crash.
    Type outbreak_effect = exp(cots_growth_rate) / (Type(1) + exp(-(cots_pred[t-1] - cots_outbreak_threshold)));
    cots_pred[t] = cots_pred[t-1]
      + outbreak_effect * cots_pred[t-1] 
      - Type(0.5) * cots_pred[t-1] * (cots_pred[t-1]/(cots_outbreak_threshold + Type(1e-8)));

    // Likelihood for observations using lognormal (data are positive and may span orders of magnitude)
    nll -= dnorm(log(coralfast_dat[t]), log(coralfast_pred[t] + Type(1e-8)), sd_obs, true);
    nll -= dnorm(log(coralslow_dat[t]), log(coralslow_pred[t] + Type(1e-8)), sd_obs, true);
    nll -= dnorm(log(cots_dat[t]), log(cots_pred[t] + Type(1e-8)), sd_obs, true);
  }

  // Report predictions
  REPORT(coralfast_pred);
  REPORT(coralslow_pred);
  REPORT(cots_pred);

  return nll;
}
