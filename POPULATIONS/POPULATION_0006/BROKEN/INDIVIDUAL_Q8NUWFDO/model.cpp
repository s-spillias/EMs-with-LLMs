#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA SECTION
  DATA_VECTOR(Year);                  // Vector of years
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  
  // PARAMETER SECTION
  PARAMETER(r_cots);                  // Maximum per capita reproduction rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(T_crit);                  // Critical temperature threshold for enhanced COTS reproduction (°C)
  PARAMETER(T_effect);                // Effect size of temperature on COTS reproduction (dimensionless)
  PARAMETER(a_fast);                  // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Maximum growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Maximum growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(alpha_fs);                // Competition coefficient of slow-growing on fast-growing coral (dimensionless)
  PARAMETER(alpha_sf);                // Competition coefficient of fast-growing on slow-growing coral (dimensionless)
  PARAMETER(imm_effect);              // Effect size of larval immigration on COTS population (dimensionless)
  PARAMETER(sigma_cots);              // Observation error standard deviation for COTS abundance (log scale)
  PARAMETER(sigma_fast);              // Observation error standard deviation for fast-growing coral cover (log scale)
  PARAMETER(sigma_slow);              // Observation error standard deviation for slow-growing coral cover (log scale)
  
  // Parameters for improved model
  PARAMETER(recruitment_delay);       // Time lag for COTS recruitment from larval to juvenile stage (years)
  PARAMETER(outbreak_threshold);      // COTS density threshold for outbreak behavior (individuals/m^2)
  PARAMETER(density_mort);            // Density-dependent mortality coefficient for COTS at high densities
  
  // New parameters for enhanced outbreak dynamics
  PARAMETER(allee_threshold);         // COTS density below which Allee effects reduce reproduction
  PARAMETER(pred_half_sat);           // Half-saturation constant for predator functional response
  PARAMETER(pred_max_rate);           // Maximum predation rate on COTS by natural predators
  PARAMETER(coral_dep_factor);        // Dependence of COTS reproduction on coral availability
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Number of time steps
  int n_years = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_years);
  vector<Type> fast_pred(n_years);
  vector<Type> slow_pred(n_years);
  
  // Initialize with first year's observed values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // Temperature effect on COTS reproduction (smooth response)
    Type temp_diff = sst_dat(t-1) - T_crit;
    Type temp_effect = Type(1.0) + T_effect * exp(-Type(0.5) * temp_diff * temp_diff);
    
    // Total coral cover (used for coral-dependent reproduction)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1);
    
    // Coral-dependent reproduction factor (sigmoid response)
    Type coral_factor = Type(1.0) / (Type(1.0) + exp(-coral_dep_factor * (total_coral - Type(10.0))));
    
    // Allee effect (reduced reproduction at low densities)
    Type allee_effect = cots_pred(t-1) / (allee_threshold + cots_pred(t-1));
    
    // Predator functional response (Type II)
    Type predation = pred_max_rate * cots_pred(t-1) / (pred_half_sat + cots_pred(t-1));
    
    // Simple reproduction with Allee effect, temperature effect, and coral dependency
    Type logistic_term = Type(1.0) - cots_pred(t-1) / K_cots;
    logistic_term = CppAD::CondExpLt(logistic_term, Type(0.0), Type(0.0), logistic_term);
    Type reproduction = r_cots * cots_pred(t-1) * allee_effect * temp_effect * coral_factor * logistic_term;
    
    // Density-dependent mortality (increases at high densities)
    Type density_mortality = density_mort * cots_pred(t-1) * cots_pred(t-1) / K_cots;
    
    // Base mortality
    Type base_mortality = m_cots * cots_pred(t-1);
    
    // Immigration effect (scaled by parameter)
    Type immigration = imm_effect * cotsimm_dat(t-1);
    
    // Update COTS population
    cots_pred(t) = cots_pred(t-1) + reproduction - base_mortality - density_mortality - predation + immigration;
    
    // Ensure non-negative population with a small minimum value
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), Type(0.01), Type(0.01), cots_pred(t));
    
    // Functional response for COTS predation on coral (Type II)
    Type denominator = Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1);
    Type consumption_fast = a_fast * cots_pred(t-1) * fast_pred(t-1) / denominator;
    Type consumption_slow = a_slow * cots_pred(t-1) * slow_pred(t-1) / denominator;
    
    // Fast-growing coral dynamics with competition
    Type competition_fast = Type(1.0) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast;
    competition_fast = CppAD::CondExpLt(competition_fast, Type(0.0), Type(0.0), competition_fast);
    Type fast_growth = r_fast * fast_pred(t-1) * competition_fast;
    fast_pred(t) = fast_pred(t-1) + fast_growth - consumption_fast;
    
    // Ensure non-negative coral cover
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(0.01), Type(0.01), fast_pred(t));
    
    // Slow-growing coral dynamics with competition
    Type competition_slow = Type(1.0) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow;
    competition_slow = CppAD::CondExpLt(competition_slow, Type(0.0), Type(0.0), competition_slow);
    Type slow_growth = r_slow * slow_pred(t-1) * competition_slow;
    slow_pred(t) = slow_pred(t-1) + slow_growth - consumption_slow;
    
    // Ensure non-negative coral cover
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(0.01), Type(0.01), slow_pred(t));
    
    // Add outbreak dynamics - rapid increase when above threshold
    Type is_outbreak = CppAD::CondExpGt(cots_pred(t-1), outbreak_threshold, Type(1.0), Type(0.0));
    if (is_outbreak > Type(0.5)) {
      // Increase growth rate during outbreaks
      Type outbreak_factor = Type(1.0) + Type(2.0) * (cots_pred(t-1) - outbreak_threshold) / outbreak_threshold;
      cots_pred(t) *= outbreak_factor;
      
      // But also increase mortality at very high densities
      Type is_high_density = CppAD::CondExpGt(cots_pred(t), K_cots * Type(0.8), Type(1.0), Type(0.0));
      if (is_high_density > Type(0.5)) {
        Type crash_factor = Type(1.0) - Type(0.5) * (cots_pred(t) - K_cots * Type(0.8)) / (K_cots * Type(0.2));
        crash_factor = CppAD::CondExpLt(crash_factor, Type(0.1), Type(0.1), crash_factor);
        cots_pred(t) *= crash_factor;
      }
    }
  }
  
  // Calculate negative log-likelihood
  for (int t = 0; t < n_years; t++) {
    // Add observation error for COTS abundance
    if (!R_IsNA(asDouble(cots_dat(t))) && cots_dat(t) > Type(0.0)) {
      nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    }
    
    // Add observation error for fast-growing coral cover
    if (!R_IsNA(asDouble(fast_dat(t))) && fast_dat(t) > Type(0.0)) {
      nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
    }
    
    // Add observation error for slow-growing coral cover
    if (!R_IsNA(asDouble(slow_dat(t))) && slow_dat(t) > Type(0.0)) {
      nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
    }
  }
  
  // Report model predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
