#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DATA INPUTS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // These are the data vectors that TMB will read in.
  
  DATA_VECTOR(Year);          // The years of the observations, for reference.
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m^2).
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%).
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%).
  DATA_VECTOR(sst_dat);       // Sea-Surface Temperature forcing data (Celsius).
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate forcing data (individuals/m^2/year).

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // PARAMETERS
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // These are the parameters that TMB will optimize.
  
  // COTS parameters
  PARAMETER(e_cots);            // COTS assimilation efficiency and conversion from % coral to COTS density ((ind/m^2)/%).
  PARAMETER(c_max_fast);        // Maximum consumption rate of fast corals per COTS (% * m^2 / (ind * year)).
  PARAMETER(k_h_fast);          // Half-saturation constant for COTS predation on fast corals (%).
  PARAMETER(c_max_slow);        // Maximum consumption rate of slow corals per COTS (% * m^2 / (ind * year)).
  PARAMETER(k_h_slow);          // Half-saturation constant for COTS predation on slow corals (%).
  PARAMETER(m_cots);            // COTS natural mortality rate at reference temperature (year^-1).
  PARAMETER(sst_m_cots_effect); // Linear effect of SST on COTS mortality (Celsius^-1).
  PARAMETER(T_ref_cots_mort);   // Reference SST for baseline COTS mortality (Celsius).

  // Fast-growing coral parameters
  PARAMETER(r_fast);            // Intrinsic growth rate of fast-growing corals (year^-1).
  PARAMETER(K_fast);            // Carrying capacity of fast-growing corals (%).
  PARAMETER(comp_fs);           // Competition effect of slow corals on fast corals (unitless).
  PARAMETER(T_opt_fast);        // Optimal SST for fast coral growth (Celsius).
  PARAMETER(T_std_fast);        // SST tolerance for fast corals (Celsius).

  // Slow-growing coral parameters
  PARAMETER(r_slow);            // Intrinsic growth rate of slow-growing corals (year^-1).
  PARAMETER(K_slow);            // Carrying capacity of slow-growing corals (%).
  PARAMETER(comp_sf);           // Competition effect of fast corals on slow corals (unitless).
  PARAMETER(T_opt_slow);        // Optimal SST for slow coral growth (Celsius).
  PARAMETER(T_std_slow);        // SST tolerance for slow corals (Celsius).

  // Observation error parameters
  PARAMETER(log_sd_cots);       // Log of the standard deviation for the COTS lognormal error model.
  PARAMETER(log_sd_fast);       // Log of the standard deviation for the fast coral lognormal error model.
  PARAMETER(log_sd_slow);       // Log of the standard deviation for the slow coral lognormal error model.

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // MODEL SETUP
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  int n = Year.size(); // Number of time steps in the data.
  
  // Create vectors to store model predictions.
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize the prediction vectors with the first observed data point.
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Initialize the negative log-likelihood (nll). This is the objective function to minimize.
  Type nll = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // DYNAMIC MODEL
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Loop over the time series, starting from the second observation.
  
  for (int t = 1; t < n; ++t) {
    // --- Environmental Effects (t-1) ---
    Type temp_effect_fast = exp(Type(-0.5) * pow((sst_dat(t-1) - T_opt_fast) / T_std_fast, 2));
    Type temp_effect_slow = exp(Type(-0.5) * pow((sst_dat(t-1) - T_opt_slow) / T_std_slow, 2));
    
    // --- Coral Dynamics (t) ---
    // 1. Logistic growth of fast-growing corals, including competition from slow corals and temperature effects.
    Type fast_growth = r_fast * temp_effect_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + comp_fs * slow_pred(t-1)) / (K_fast + Type(1e-8)));
    
    // 2. Logistic growth of slow-growing corals, including competition from fast corals and temperature effects.
    Type slow_growth = r_slow * temp_effect_slow * slow_pred(t-1) * (Type(1.0) - (slow_pred(t-1) + comp_sf * fast_pred(t-1)) / (K_slow + Type(1e-8)));

    // --- Predation Dynamics (t) ---
    // 3. Predation loss of fast corals due to COTS, using a Michaelis-Menten functional response.
    Type predation_loss_fast = cots_pred(t-1) * c_max_fast * fast_pred(t-1) / (k_h_fast + fast_pred(t-1) + Type(1e-8));
    
    // 4. Predation loss of slow corals due to COTS.
    Type predation_loss_slow = cots_pred(t-1) * c_max_slow * slow_pred(t-1) / (k_h_slow + slow_pred(t-1) + Type(1e-8));

    // --- COTS Dynamics (t) ---
    // 5. COTS growth based on assimilated coral biomass.
    Type cots_growth = e_cots * (predation_loss_fast + predation_loss_slow);
    
    // 6. COTS mortality, influenced by sea surface temperature.
    Type cots_mortality = m_cots * (Type(1.0) + sst_m_cots_effect * (sst_dat(t-1) - T_ref_cots_mort)) * cots_pred(t-1);

    // --- State Variable Updates ---
    // Update state variables using an explicit Euler step (dt=1 year).
    // Prevent negative population sizes using a ternary operator, which is AD-safe with TMB.
    Type next_fast = fast_pred(t-1) + fast_growth - predation_loss_fast;
    fast_pred(t) = (next_fast > Type(0.0)) ? next_fast : Type(0.0);

    Type next_slow = slow_pred(t-1) + slow_growth - predation_loss_slow;
    slow_pred(t) = (next_slow > Type(0.0)) ? next_slow : Type(0.0);

    Type next_cots = cots_pred(t-1) + cots_growth - cots_mortality + cotsimm_dat(t-1);
    cots_pred(t) = (next_cots > Type(0.0)) ? next_cots : Type(0.0);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // LIKELIHOOD CALCULATION
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Compare model predictions to observed data to calculate the likelihood.
    // A lognormal error distribution is used as abundances and cover are strictly positive.
    
    // Add a small constant to predictions to prevent log(0).
    Type cots_pred_safe = cots_pred(t) + Type(1e-8);
    Type fast_pred_safe = fast_pred(t) + Type(1e-8);
    Type slow_pred_safe = slow_pred(t) + Type(1e-8);

    // Calculate standard deviations from log-transformed parameters.
    Type sd_cots = exp(log_sd_cots);
    Type sd_fast = exp(log_sd_fast);
    Type sd_slow = exp(log_sd_slow);

    // Add the negative log-likelihood contribution for each state variable at time t.
    nll -= dnorm(log(cots_dat(t)), log(cots_pred_safe), sd_cots, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred_safe), sd_fast, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred_safe), sd_slow, true);
  }

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // REPORTING
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Report the predicted time series for analysis and visualization.
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
EQUATION DESCRIPTIONS:
1. Fast Coral Growth: d(fast)/dt = r_fast * temp_effect_fast * fast * (1 - (fast + comp_fs * slow) / K_fast)
   - Logistic growth of fast-growing corals, reduced by competition from slow corals and modulated by sea surface temperature.

2. Slow Coral Growth: d(slow)/dt = r_slow * temp_effect_slow * slow * (1 - (slow + comp_sf * fast) / K_slow)
   - Logistic growth of slow-growing corals, reduced by competition from fast corals and modulated by sea surface temperature.

3. Fast Coral Predation Loss: Loss_fast = cots * c_max_fast * fast / (k_h_fast + fast)
   - Consumption of fast-growing corals by COTS, following a Michaelis-Menten (saturating) functional response.

4. Slow Coral Predation Loss: Loss_slow = cots * c_max_slow * slow / (k_h_slow + slow)
   - Consumption of slow-growing corals by COTS.

5. COTS Growth: Growth_cots = e_cots * (Loss_fast + Loss_slow)
   - COTS population growth is proportional to the total biomass of coral consumed, converted by an efficiency factor.

6. COTS Mortality: Mortality_cots = m_cots * (1 + sst_m_cots_effect * (sst - T_ref_cots_mort)) * cots
   - Natural mortality of COTS, with a baseline rate that is linearly adjusted by deviations from a reference sea surface temperature.

7. COTS Immigration: Immigration_cots = cotsimm_dat
   - External input of COTS larvae, acting as a driver for population outbreaks.
*/
