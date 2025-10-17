#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DATA INPUTS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // These are the data inputs from the CSV files.
  DATA_VECTOR(Year);          // Vector of years for the time series
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m^2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Observed sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // Observed COTS larval immigration (individuals/m^2/year)

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // MODEL PARAMETERS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // These parameters are estimated by the model.
  PARAMETER(r_fast);          // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(r_slow);          // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(K_coral);         // Total carrying capacity for all corals (%)
  PARAMETER(sst_opt_coral);   // Optimal sea-surface temperature for coral growth (Celsius)
  PARAMETER(sst_width_coral); // Width of the SST tolerance curve for coral growth (Celsius)
  PARAMETER(a_fast);          // COTS attack rate on fast-growing corals (m^2/individual/year)
  PARAMETER(a_slow);          // COTS attack rate on slow-growing corals (m^2/individual/year)
  PARAMETER(h_cots);          // COTS handling time for coral (year*%_cover^-1)
  PARAMETER(e_fast);          // COTS assimilation efficiency for fast-growing corals (dimensionless)
  PARAMETER(e_slow);          // COTS assimilation efficiency for slow-growing corals (dimensionless)
  PARAMETER(m_cots);          // COTS natural mortality rate (year^-1)
  PARAMETER(m_dd_cots);       // COTS density-dependent mortality coefficient ((individuals/m^2)^-1 * year^-1)
  PARAMETER(log_sd_cots);     // Log of the standard deviation for COTS density observations
  PARAMETER(log_sd_fast);     // Log of the standard deviation for fast-growing coral cover observations
  PARAMETER(log_sd_slow);     // Log of the standard deviation for slow-growing coral cover observations

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // TRANSFORM PARAMETERS & INITIALIZE VARIABLES
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Calculate the number of time steps in the data
  int n_timesteps = Year.size();

  // Transform log-standard deviations to standard deviations
  Type sd_cots = exp(log_sd_cots);
  Type sd_fast = exp(log_sd_fast);
  Type sd_slow = exp(log_sd_slow);

  // Initialize prediction vectors
  vector<Type> cots_pred(n_timesteps);
  vector<Type> fast_pred(n_timesteps);
  vector<Type> slow_pred(n_timesteps);

  // Set initial conditions for predictions from the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize the negative log-likelihood
  Type nll = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // MODEL EQUATION DESCRIPTIONS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // 1. Fast Coral Growth: Logistic growth limited by total coral cover, modulated by SST.
  //    fast_growth = r_fast * fast_cover * (1 - (fast_cover + slow_cover) / K_coral) * sst_effect
  // 2. Slow Coral Growth: Logistic growth limited by total coral cover, modulated by SST.
  //    slow_growth = r_slow * slow_cover * (1 - (fast_cover + slow_cover) / K_coral) * sst_effect
  // 3. SST Effect on Coral Growth: A Gaussian function where growth peaks at sst_opt_coral.
  //    sst_effect = exp(-0.5 * ((sst - sst_opt_coral) / sst_width_coral)^2)
  // 4. COTS Predation on Corals: A multi-species Holling Type II functional response.
  //    predation_denominator = 1 + a_fast * h_cots * fast_cover + a_slow * h_cots * slow_cover
  //    fast_predation = (a_fast * fast_cover * cots_density) / predation_denominator
  //    slow_predation = (a_slow * slow_cover * cots_density) / predation_denominator
  // 5. COTS Population Growth: Based on assimilated coral biomass from predation.
  //    cots_growth = e_fast * fast_predation + e_slow * slow_predation
  // 6. COTS Population Decline: Includes natural and density-dependent mortality.
  //    cots_mortality = m_cots * cots_density + m_dd_cots * cots_density^2
  // 7. COTS Immigration: External larval supply from data.
  //    cots_immigration = cotsimm_dat
  // 8. State Dynamics (Euler integration, dt=1 year):
  //    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation
  //    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation
  //    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // PROCESS MODEL (TIME-STEPPING LOOP)
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  for (int i = 1; i < n_timesteps; ++i) {
    // Use a small constant to prevent division by zero or log of zero.
    Type epsilon = 1e-8;

    // Get values from the previous time step
    Type cots_prev = cots_pred(i-1);
    Type fast_prev = fast_pred(i-1);
    Type slow_prev = slow_pred(i-1);

    // --- Coral Dynamics ---
    // 1. Calculate total coral cover from previous step
    Type total_coral_prev = fast_prev + slow_prev;

    // 2. Calculate SST effect on coral growth
    Type sst_effect = exp(Type(-0.5) * pow((sst_dat(i-1) - sst_opt_coral) / sst_width_coral, 2));

    // 3. Calculate growth for fast-growing corals
    Type fast_growth = r_fast * fast_prev * (Type(1.0) - total_coral_prev / K_coral) * sst_effect;

    // 4. Calculate growth for slow-growing corals
    Type slow_growth = r_slow * slow_prev * (Type(1.0) - total_coral_prev / K_coral) * sst_effect;

    // --- COTS Predation Dynamics ---
    // 5. Calculate the denominator for the Holling Type II functional response
    Type predation_denominator = Type(1.0) + a_fast * h_cots * fast_prev + a_slow * h_cots * slow_prev;

    // 6. Calculate predation on fast-growing corals
    Type fast_predation = (a_fast * fast_prev * cots_prev) / (predation_denominator + epsilon);

    // 7. Calculate predation on slow-growing corals
    Type slow_predation = (a_slow * slow_prev * cots_prev) / (predation_denominator + epsilon);

    // --- COTS Population Dynamics ---
    // 8. Calculate COTS growth from assimilated coral
    Type cots_growth = e_fast * fast_predation + e_slow * slow_predation;

    // 9. Calculate COTS mortality (natural + density-dependent)
    Type cots_mortality = m_cots * cots_prev + m_dd_cots * cots_prev * cots_prev;

    // 10. Get COTS immigration for the current step
    Type cots_immigration = cotsimm_dat(i-1);

    // --- Update State Variables using Euler method (dt=1 year) ---
    // Update fast coral cover
    fast_pred(i) = fast_prev + fast_growth - fast_predation;
    // Ensure cover is non-negative
    if (fast_pred(i) < 0) fast_pred(i) = 0;

    // Update slow coral cover
    slow_pred(i) = slow_prev + slow_growth - slow_predation;
    // Ensure cover is non-negative
    if (slow_pred(i) < 0) slow_pred(i) = 0;

    // Update COTS density
    cots_pred(i) = cots_prev + cots_growth - cots_mortality + cots_immigration;
    // Ensure density is non-negative
    if (cots_pred(i) < 0) cots_pred(i) = 0;
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // LIKELIHOOD CALCULATION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Compare model predictions with observed data to calculate the likelihood of the parameters.
  // We use a lognormal error distribution, which is equivalent to a normal distribution on log-transformed data.
  for (int i = 0; i < n_timesteps; ++i) {
    // Add a small constant to prevent log(0)
    Type epsilon = 1e-8;

    // COTS likelihood
    nll -= dnorm(log(cots_dat(i) + epsilon), log(cots_pred(i) + epsilon), sd_cots, true);

    // Fast coral likelihood
    nll -= dnorm(log(fast_dat(i) + epsilon), log(fast_pred(i) + epsilon), sd_fast, true);

    // Slow coral likelihood
    nll -= dnorm(log(slow_dat(i) + epsilon), log(slow_pred(i) + epsilon), sd_slow, true);
  }
  
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // PARAMETER BOUND PENALTIES
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Add penalties to the nll if parameters go outside their plausible biological ranges.
  // This helps guide the optimization process.
  // Example for one parameter (can be extended for others):
  // if (r_fast < 0.0) { nll -= dnorm(r_fast, Type(0.0), Type(0.1), true) - dnorm(Type(0.0), Type(0.0), Type(0.1), true); }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // REPORTING SECTION
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Report predicted values for plotting and analysis.
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
