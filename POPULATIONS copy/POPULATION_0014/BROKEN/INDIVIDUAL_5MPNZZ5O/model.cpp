#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  // These are the time series data provided to the model
  DATA_VECTOR(Year);          // Vector of years for the time series
  DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Observed Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // Observed COTS larval immigration (individuals/m2/year)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  // These are the values the model will optimize
  // COTS parameters
  PARAMETER(alpha_fast);      // COTS attack rate on fast-growing corals (m2/individual/year)
  PARAMETER(alpha_slow);      // COTS attack rate on slow-growing corals (m2/individual/year)
  PARAMETER(h_cots);          // COTS handling time for consuming coral (% coral/year)
  PARAMETER(assim_eff);       // COTS assimilation efficiency (dimensionless)
  PARAMETER(mort_cots);       // COTS natural mortality rate (year^-1)
  PARAMETER(sst_opt_cots);    // Optimal SST for COTS growth (Celsius)
  PARAMETER(sst_width_cots);  // SST tolerance/range for COTS growth (Celsius)

  // Coral parameters
  PARAMETER(r_fast);          // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(r_slow);          // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(K_coral);         // Carrying capacity for total coral cover (%)
  PARAMETER(bleach_mort_max); // Maximum bleaching mortality rate for corals (year^-1)
  PARAMETER(bleach_sensitivity); // Sensitivity of coral bleaching to SST increase (Celsius^-1)
  PARAMETER(sst_bleach_thresh);  // SST threshold for coral bleaching (Celsius)

  // Standard deviations for the likelihood calculation
  PARAMETER(log_sd_cots);     // Log of the standard deviation for COTS abundance
  PARAMETER(log_sd_fast);     // Log of the standard deviation for fast-growing coral cover
  PARAMETER(log_sd_slow);     // Log of the standard deviation for slow-growing coral cover

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------
  int n_steps = Year.size(); // Number of time steps in the data

  // Transform log standard deviations to positive values and add a minimum value for stability
  Type sd_cots = exp(log_sd_cots) + Type(0.01);
  Type sd_fast = exp(log_sd_fast) + Type(0.01);
  Type sd_slow = exp(log_sd_slow) + Type(0.01);

  // Create vectors to store model predictions
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);

  // Initialize predictions with the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize the negative log-likelihood
  Type nll = 0.0;

  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  // This model simulates the ecosystem dynamics over time.
  // 1. COTS predation on corals follows a multi-prey Holling Type II functional response.
  // 2. Energy gained from predation is converted to COTS biomass (growth) via an assimilation efficiency.
  // 3. COTS growth is modulated by sea-surface temperature (SST), with an optimal temperature range.
  // 4. The COTS population changes based on growth, natural mortality, and external larval immigration.
  // 5. Corals (both fast- and slow-growing) grow logistically, competing for available space.
  // 6. Coral populations are reduced by COTS predation.
  // 7. High SST causes coral bleaching, modeled as an additional source of mortality using a logistic function.
  // ------------------------------------------------------------------------

  // Loop over time, starting from the second time step
  for (int t = 1; t < n_steps; ++t) {
    // Get state variables from the previous time step (t-1)
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);

    // --- COTS Predation on Corals ---
    // Holling Type II functional response denominator
    Type fr_denom = Type(1.0) + alpha_fast * h_cots * fast_prev + alpha_slow * h_cots * slow_prev;
    // Consumption of fast-growing corals
    Type consumption_fast = cots_prev * alpha_fast * fast_prev / (fr_denom + Type(1e-8));
    // Consumption of slow-growing corals
    Type consumption_slow = cots_prev * alpha_slow * slow_prev / (fr_denom + Type(1e-8));

    // --- COTS Population Dynamics ---
    // Temperature effect on COTS growth (Gaussian function)
    Type sst_effect_cots = exp(-Type(0.5) * pow((sst_dat(t-1) - sst_opt_cots) / sst_width_cots, 2));
    // Total energy gain from predation, modulated by temperature
    Type cots_gain = assim_eff * (consumption_fast + consumption_slow) * sst_effect_cots;
    // Natural mortality
    Type cots_loss = mort_cots * cots_prev;
    // Update COTS prediction
    cots_pred(t) = cots_prev + cots_gain - cots_loss + cotsimm_dat(t-1);

    // --- Coral Population Dynamics ---
    // Available space for coral growth (as a fraction of total capacity)
    Type space_available = Type(1.0) - (fast_prev + slow_prev) / (K_coral + Type(1e-8));
    // Temperature-induced bleaching mortality (logistic function)
    Type bleaching_effect = bleach_mort_max / (Type(1.0) + exp(-bleach_sensitivity * (sst_dat(t-1) - sst_bleach_thresh)));
    
    // Fast-growing coral dynamics
    Type fast_growth = r_fast * fast_prev * space_available;
    Type fast_bleaching_loss = fast_prev * bleaching_effect;
    fast_pred(t) = fast_prev + fast_growth - consumption_fast - fast_bleaching_loss;

    // Slow-growing coral dynamics
    Type slow_growth = r_slow * slow_prev * space_available;
    Type slow_bleaching_loss = slow_prev * bleaching_effect;
    slow_pred(t) = slow_prev + slow_growth - consumption_slow - slow_bleaching_loss;

    // --- Numerical Stability ---
    // Ensure predictions do not become negative
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));

    // ------------------------------------------------------------------------
    // LIKELIHOOD CALCULATION
    // ------------------------------------------------------------------------
    // Compare model predictions with observed data using a lognormal distribution
    // This assumes that the logarithm of the data is normally distributed
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sd_cots, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sd_fast, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sd_slow, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  // Report predicted time series and the final negative log-likelihood
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(nll);

  return nll;
}
