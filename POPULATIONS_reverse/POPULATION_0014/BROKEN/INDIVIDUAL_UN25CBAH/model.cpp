#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA
  DATA_VECTOR(Year);                  // Years of observation
  DATA_VECTOR(sst_dat);               // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);           // COTS immigration rate (individuals/m^2/year)
  DATA_VECTOR(cots_dat);              // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);              // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);              // Observed fast-growing coral cover (%)
  
  // PARAMETERS
  PARAMETER(r_cots);                  // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);                  // Carrying capacity of COTS (individuals/m^2)
  PARAMETER(m_cots);                  // Natural mortality rate of COTS (year^-1)
  PARAMETER(a_fast);                  // Attack rate on fast-growing coral (m^2/individual/year)
  PARAMETER(a_slow);                  // Attack rate on slow-growing coral (m^2/individual/year)
  PARAMETER(h_fast);                  // Handling time for fast-growing coral (year/% cover)
  PARAMETER(h_slow);                  // Handling time for slow-growing coral (year/% cover)
  PARAMETER(r_fast);                  // Intrinsic growth rate of fast-growing coral (year^-1)
  PARAMETER(r_slow);                  // Intrinsic growth rate of slow-growing coral (year^-1)
  PARAMETER(K_fast);                  // Carrying capacity of fast-growing coral (% cover)
  PARAMETER(K_slow);                  // Carrying capacity of slow-growing coral (% cover)
  PARAMETER(alpha_fs);                // Competition coefficient: effect of slow on fast coral
  PARAMETER(alpha_sf);                // Competition coefficient: effect of fast on slow coral
  PARAMETER(temp_opt);                // Optimal temperature for coral growth (°C)
  PARAMETER(temp_tol);                // Temperature tolerance range (°C)
  PARAMETER(imm_effect);              // Effect of immigration on COTS population
  PARAMETER(coral_threshold);         // Coral cover threshold for COTS survival (% cover)
  PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
  PARAMETER(sigma_slow);              // Observation error SD for slow-growing coral (log scale)
  PARAMETER(sigma_fast);              // Observation error SD for fast-growing coral (log scale)
  PARAMETER(eff_max);                 // Maximum multiplier for COTS predation efficiency during outbreaks
  PARAMETER(eff_half);                // COTS density at which predation efficiency reaches half maximum
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-3);
  
  // Number of time steps
  int n_steps = Year.size();
  
  // Vectors to store model predictions
  vector<Type> cots_pred(n_steps);
  vector<Type> slow_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series simulation
  for (int t = 1; t < n_steps; t++) {
    // 1. Calculate temperature effect on coral growth (Gaussian response curve)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-0.5 * temp_diff * temp_diff / (temp_tol * temp_tol + eps));
    
    // 2. Calculate total coral cover (food availability for COTS)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    
    // 3. Calculate density-dependent predation efficiency multiplier
    // This represents increased feeding efficiency at higher COTS densities due to group feeding behavior
    Type eff_multiplier = 1.0 + (eff_max - 1.0) * cots_pred(t-1) / (eff_half + cots_pred(t-1) + eps);
    
    // 4. Calculate functional responses for COTS feeding on corals (Type II)
    Type a_fast_eff = a_fast * eff_multiplier;
    Type a_slow_eff = a_slow * eff_multiplier;
    
    // Calculate denominator for functional response
    Type denom = 1.0 + a_fast_eff * h_fast * fast_pred(t-1) + a_slow_eff * h_slow * slow_pred(t-1);
    denom = max(denom, eps);  // Ensure denominator is not too small
    
    // Calculate feeding rates
    Type F_fast = (a_fast_eff * fast_pred(t-1)) / denom;
    Type F_slow = (a_slow_eff * slow_pred(t-1)) / denom;
    
    // 5. Calculate food limitation effect on COTS (smooth transition at threshold)
    Type food_diff = total_coral - coral_threshold;
    Type food_limitation = 1.0 / (1.0 + exp(-5.0 * food_diff));
    
    // 6. COTS population dynamics
    Type density_effect = 1.0 - cots_pred(t-1) / (K_cots + eps);
    density_effect = max(density_effect, -1.0);  // Bound density effect
    
    Type cots_growth = r_cots * cots_pred(t-1) * density_effect * food_limitation;
    Type cots_mortality = m_cots * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    // Update COTS population
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = max(cots_pred(t), eps);  // Ensure positive values
    
    // 7. Coral dynamics with competition and COTS predation
    // Fast-growing coral
    Type fast_competition = 1.0 - (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / (K_fast + eps);
    fast_competition = max(fast_competition, -1.0);  // Bound competition term
    
    Type fast_growth = r_fast * fast_pred(t-1) * fast_competition * temp_effect;
    Type fast_predation = F_fast * cots_pred(t-1);
    
    // Update fast-growing coral
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    fast_pred(t) = max(fast_pred(t), eps);  // Ensure positive values
    fast_pred(t) = min(fast_pred(t), K_fast);  // Ensure below carrying capacity
    
    // Slow-growing coral
    Type slow_competition = 1.0 - (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / (K_slow + eps);
    slow_competition = max(slow_competition, -1.0);  // Bound competition term
    
    Type slow_growth = r_slow * slow_pred(t-1) * slow_competition * temp_effect;
    Type slow_predation = F_slow * cots_pred(t-1);
    
    // Update slow-growing coral
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    slow_pred(t) = max(slow_pred(t), eps);  // Ensure positive values
    slow_pred(t) = min(slow_pred(t), K_slow);  // Ensure below carrying capacity
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type min_sigma = Type(0.1);  // Minimum sigma for better numerical stability
  
  for (int t = 0; t < n_steps; t++) {
    // Use max to ensure sigma is not too small
    Type sigma_cots_t = max(sigma_cots, min_sigma);
    Type sigma_slow_t = max(sigma_slow, min_sigma);
    Type sigma_fast_t = max(sigma_fast, min_sigma);
    
    // Calculate log-likelihood contributions
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_t, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_t, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_t, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
