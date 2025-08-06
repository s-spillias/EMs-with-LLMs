#include <TMB.hpp>

// 1. Data:
//    - Year: Time steps from the data file.
//    - sst_dat: Sea Surface Temperature data (°C).
//    - cotsimm_dat: Crown-of-thorns larval immigration rate (individuals/m2/year).
//    - cots_dat: Adult COTS abundance (individuals/m2).
//    - fast_dat: Fast-growing coral cover (Acropora spp.) in %.
//    - slow_dat: Slow-growing coral cover (Faviidae spp. and Porities spp.) in %.
//
// 2. Parameters and equations:
//    (1) COTS outbreak dynamics:
//        cots_pred[t] = cots_pred[t-1] +
//          [ growth_rate * cots_pred[t-1] * ( (fast_dat[t-1]+slow_dat[t-1])/(fast_dat[t-1]+slow_dat[t-1]+saturation) ) 
//            - decline_rate * cots_pred[t-1] ] * dt
//    (2) Environmental modification through sea surface temperature is embedded in the outbreak growth.
//    (3) Smooth transitions and small constants (e.g., 1e-8) are used to avoid division by zero.
//    (4) Only previous time step values are used in predictions to avoid data leakage.
//
// 3. Likelihood:
//    - Observations (cots_dat) are assumed to follow a lognormal distribution around the predictions.
//    - A fixed minimum standard deviation is used for numerical stability.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs from file
  DATA_VECTOR(Year);                // Time (years)
  DATA_VECTOR(sst_dat);             // Sea Surface Temperature (°C)
  DATA_VECTOR(cotsimm_dat);         // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);            // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);            // Fast-growing coral cover (Acropora spp.) in %
  DATA_VECTOR(slow_dat);            // Slow-growing coral cover (Faviidae spp. and Porities spp.) in %

  int n = Year.size();              // Number of time steps

  // Parameters (log-scale parameters ensure positivity)
  PARAMETER(log_growth_rate);       // Log of intrinsic outbreak growth rate (log(year^-1))
  PARAMETER(log_decline_rate);      // Log of outbreak decline rate (log(year^-1))
  PARAMETER(log_threshold);         // Log of threshold resource level triggering outbreak (log(units))
  PARAMETER(efficiency_fast);       // Efficiency factor for predation on fast-growing corals (unitless)
  PARAMETER(efficiency_slow);       // Efficiency factor for predation on slow-growing corals (unitless)
  PARAMETER(effect_sst);            // Effect of sea surface temperature on outbreak progression (per °C)
  PARAMETER(quad_effect_sst);       // Quadratic effect of sea surface temperature on outbreak progression (per °C^2)
  PARAMETER(log_saturation);        // Log of saturation constant for resource limitation (log(units))

  // Parameter transformations to ensure positivity where applicable
  Type growth_rate = exp(log_growth_rate);    // Intrinsic growth rate (year^-1)
  Type decline_rate = exp(log_decline_rate);    // Decline rate during bust (year^-1)
  Type threshold    = exp(log_threshold);       // Threshold resource level (units)
  Type saturation   = exp(log_saturation);      // Saturation constant (units)

  // Likelihood accumulation
  Type nll = 0.0;

  // Predicted state vectors for adult COTS numbers and coral covers
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  cots_pred[0] = cots_dat(0);   // Initialize COTS with first observed value
  fast_pred[0] = fast_dat(0);    // Initialize fast-growing coral with first observed value
  slow_pred[0] = slow_dat(0);    // Initialize slow-growing coral with first observed value

  for(int t = 1; t < n; t++){
    // 1. Calculate total coral cover from previous time step with a small constant to prevent division by zero.
    Type coral_total = fast_dat(t-1) + slow_dat(t-1) + Type(1e-8);
    // 2. Resource limitation modeled as a saturating function.
    Type resource_limitation = coral_total / (coral_total + saturation);

    // 3. Time difference between measurements
    Type dt = Year(t) - Year(t-1);
    // Introduce environmental modification: Sea Surface Temperature effect modulates outbreak growth rate.
    Type temperature_factor = 1 + effect_sst * sst_dat(t-1) + quad_effect_sst * sst_dat(t-1) * sst_dat(t-1);

    // 4. COTS outbreak dynamics:
    //    Equation Explanation:
    //     (a) Growth is proportional to current COTS numbers, resource availability, and intrinsic growth rate.
    //     (b) Decline is modeled as a proportional loss with decline_rate.
    cots_pred[t] = cots_pred[t-1] +
                   (growth_rate * cots_pred[t-1] * resource_limitation * temperature_factor - decline_rate * cots_pred[t-1]) * dt;

    // 5. Ensure numerical stability by preventing negative predictions:
    cots_pred[t] = CppAD::CondExpGt(cots_pred[t], Type(1e-8), cots_pred[t], Type(1e-8));
    
    // 5a. Coral dynamics:
    // Fast-growing coral dynamics
    Type fast_growth_rate = Type(0.2);  // (year^-1), assumed constant growth rate
    Type fast_cap = Type(100.0);        // (%) maximum coral cover
    fast_pred[t] = fast_pred[t-1] + dt * ( fast_growth_rate * fast_pred[t-1] * (1 - fast_pred[t-1] / fast_cap)
                         - efficiency_fast * cots_pred[t-1] * fast_pred[t-1] );
                         
    // Slow-growing coral dynamics
    Type slow_growth_rate = Type(0.1);  // (year^-1), assumed constant growth rate
    Type slow_cap = Type(100.0);        // (%) maximum coral cover
    slow_pred[t] = slow_pred[t-1] + dt * ( slow_growth_rate * slow_pred[t-1] * (1 - slow_pred[t-1] / slow_cap)
                         - efficiency_slow * cots_pred[t-1] * slow_pred[t-1] );

    // 6. Likelihood contribution:
    //    Observations cots_dat are assumed lognormally distributed about predictions.
    //    A fixed standard deviation sigma = 0.1 is used for numerical robustness.
    Type sigma = 0.1;
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred[t] + Type(1e-8)), sigma, true);
    
    // Likelihood contributions for coral observations
    Type sigma_coral = 0.1;
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), sigma_coral, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), sigma_coral, true);
  }

  // Report predictions for external diagnostics.
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
