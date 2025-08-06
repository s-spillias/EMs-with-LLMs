#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS larval immigration rate (individuals/m^2/year)
  
  // PARAMETERS
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS population (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS population (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(K_fast);                  // Maximum cover of fast-growing coral (%)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_slow);                  // Maximum cover of slow-growing coral (%)
  PARAMETER(a_fast);                  // Attack rate of COTS on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate of COTS on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for COTS feeding on fast-growing coral (% cover)
  PARAMETER(h_slow);                  // Handling time for COTS feeding on slow-growing coral (% cover)
  PARAMETER(temp_opt);                // Optimal temperature for COTS recruitment (°C)
  PARAMETER(temp_width);              // Temperature range width for COTS recruitment (°C)
  PARAMETER(imm_effect);              // Effect of larval immigration on COTS recruitment (dimensionless)
  PARAMETER(competition);             // Competition coefficient between coral types (dimensionless)
  PARAMETER(bleach_threshold);        // Temperature threshold for coral bleaching (°C)
  PARAMETER(bleach_mortality_fast);   // Mortality rate of fast-growing coral during bleaching (year^-1)
  PARAMETER(bleach_mortality_slow);   // Mortality rate of slow-growing coral during bleaching (year^-1)
  PARAMETER(sigma_cots);              // Observation error standard deviation for COTS abundance (log scale)
  PARAMETER(sigma_fast);              // Observation error standard deviation for fast-growing coral cover (log scale)
  PARAMETER(sigma_slow);              // Observation error standard deviation for slow-growing coral cover (log scale)
  PARAMETER(allee_threshold);         // Population density threshold for Allee effect in COTS (individuals/m^2)
  PARAMETER(allee_strength);          // Strength of the Allee effect in COTS population (dimensionless)
  
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
  
  // Initialize with first year's data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sigma = Type(0.01);
  Type sigma_cots_adj = sigma_cots + min_sigma;
  Type sigma_fast_adj = sigma_fast + min_sigma;
  Type sigma_slow_adj = sigma_slow + min_sigma;
  
  // Time series simulation
  for (int t = 1; t < n_years; t++) {
    // Previous time step values - ensure positive values
    Type cots_t0 = cots_pred(t-1) + eps;
    Type fast_t0 = fast_pred(t-1) + eps;
    Type slow_t0 = slow_pred(t-1) + eps;
    Type sst = sst_dat(t-1);
    Type cotsimm = cotsimm_dat(t-1);
    
    // 1. Temperature effect on COTS recruitment
    // Gaussian response curve for temperature effect on COTS recruitment
    Type temp_diff = sst - temp_opt;
    Type temp_width_safe = temp_width + eps;
    Type temp_effect = exp(-0.5 * pow(temp_diff / temp_width_safe, 2));
    
    // 2. COTS functional response (Type II) for predation on corals
    // Holling Type II functional response for COTS predation
    Type denominator = 1.0 + a_fast * h_fast * fast_t0 + a_slow * h_slow * slow_t0;
    Type pred_fast = (a_fast * fast_t0 * cots_t0) / denominator;
    Type pred_slow = (a_slow * slow_t0 * cots_t0) / denominator;
    
    // 3. Bleaching effect on corals
    // Smooth transition function for bleaching effect
    Type bleach_effect = 1.0 / (1.0 + exp(-2.0 * (sst - bleach_threshold)));
    
    // 4. COTS population dynamics with density-dependent outbreak potential
    // Smooth sigmoid function for outbreak potential based on population density
    // This creates a continuous transition from normal to outbreak dynamics
    // Using sigmoid function: 1 + s*sigmoid(d*(x-t)) where s=strength, d=steepness, t=threshold
    Type sigmoid_steepness = 5.0; // Controls how sharp the transition is
    Type sigmoid_term = 1.0 / (1.0 + exp(-sigmoid_steepness * (cots_t0 - allee_threshold)));
    Type outbreak_factor = 1.0 + allee_strength * sigmoid_term;
    
    // COTS population growth with density dependence, outbreak potential, and temperature effect
    Type cots_growth = r_cots * cots_t0 * (1.0 - cots_t0 / K_cots) * temp_effect * outbreak_factor;
    
    // Immigration effect - smooth saturation function
    Type imm_term = imm_effect * cotsimm / (1.0 + cotsimm);
    
    // Food limitation effect (COTS mortality increases when coral cover is low)
    Type total_coral = fast_t0 + slow_t0;
    Type food_limitation = m_cots * (1.0 + 1.0 / total_coral);
    
    // Update COTS abundance - ensure positive values with softplus
    Type cots_new = cots_t0 + cots_growth - food_limitation * cots_t0 + imm_term;
    cots_pred(t) = log(1.0 + exp(cots_new)) - log(1.0 + 1.0); // Softplus function to ensure positivity
    
    // 5. Coral dynamics
    // Fast-growing coral dynamics with logistic growth, competition, predation, and bleaching
    Type fast_growth = r_fast * fast_t0 * (1.0 - (fast_t0 + competition * slow_t0) / K_fast);
    Type fast_bleaching = bleach_mortality_fast * bleach_effect * fast_t0;
    
    // Update fast-growing coral cover - ensure positive values with softplus
    Type fast_new = fast_t0 + fast_growth - pred_fast - fast_bleaching;
    fast_pred(t) = log(1.0 + exp(fast_new)) - log(1.0 + 1.0);
    
    // Slow-growing coral dynamics with logistic growth, competition, predation, and bleaching
    Type slow_growth = r_slow * slow_t0 * (1.0 - (slow_t0 + competition * fast_t0) / K_slow);
    Type slow_bleaching = bleach_mortality_slow * bleach_effect * slow_t0;
    
    // Update slow-growing coral cover - ensure positive values with softplus
    Type slow_new = slow_t0 + slow_growth - pred_slow - slow_bleaching;
    slow_pred(t) = log(1.0 + exp(slow_new)) - log(1.0 + 1.0);
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  for (int t = 0; t < n_years; t++) {
    // Add small constant to data and predictions to handle zeros
    Type cots_obs = cots_dat(t) + eps;
    Type cots_mod = cots_pred(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type fast_mod = fast_pred(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type slow_mod = slow_pred(t) + eps;
    
    // Log-normal likelihood for COTS abundance
    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots_adj, true);
    
    // Log-normal likelihood for fast-growing coral cover
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_fast_adj, true);
    
    // Log-normal likelihood for slow-growing coral cover
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_slow_adj, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
