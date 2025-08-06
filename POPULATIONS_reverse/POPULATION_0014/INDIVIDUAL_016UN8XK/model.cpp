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
  PARAMETER(temp_repro_threshold);    // Temperature threshold for enhanced COTS reproduction (°C)
  PARAMETER(temp_repro_effect);       // Effect of temperature on COTS reproduction (dimensionless)
  PARAMETER(sigma_cots);              // Observation error SD for COTS (log scale)
  PARAMETER(sigma_slow);              // Observation error SD for slow-growing coral (log scale)
  PARAMETER(sigma_fast);              // Observation error SD for fast-growing coral (log scale)
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
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
    // Temperature effect on coral growth (Gaussian response curve)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_tol, 2));
    
    // Total coral cover (food availability for COTS)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // Functional responses for COTS feeding on corals (Type II)
    Type denom = 1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1);
    Type F_fast = (a_fast * fast_pred(t-1)) / denom;
    Type F_slow = (a_slow * slow_pred(t-1)) / denom;
    
    // Food limitation effect on COTS (smooth transition at threshold)
    Type food_limitation = 0.1 + 0.9 / (1.0 + exp(-5.0 * (total_coral - coral_threshold)));
    
    // Temperature effect on COTS reproduction
    Type temp_effect_cots = 1.0;
    if (sst_dat(t-1) > temp_repro_threshold) {
      temp_effect_cots = 1.0 + temp_repro_effect * (sst_dat(t-1) - temp_repro_threshold) / 2.0;
    }
    
    // COTS population dynamics
    Type density_factor = std::max(Type(0.0), Type(1.0 - cots_pred(t-1) / K_cots));
    Type cots_growth = r_cots * cots_pred(t-1) * density_factor * food_limitation * temp_effect_cots;
    Type cots_mortality = m_cots * cots_pred(t-1);
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    
    cots_pred(t) = std::max(eps, cots_pred(t-1) + cots_growth - cots_mortality + cots_immigration);
    
    // Fast-growing coral dynamics
    Type competition_fast = (fast_pred(t-1) + alpha_fs * slow_pred(t-1)) / K_fast;
    competition_fast = std::min(Type(1.0), competition_fast);
    
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - competition_fast) * temp_effect;
    Type fast_predation = std::min(fast_pred(t-1), F_fast * cots_pred(t-1));
    
    fast_pred(t) = std::max(eps, fast_pred(t-1) + fast_growth - fast_predation);
    
    // Slow-growing coral dynamics
    Type competition_slow = (slow_pred(t-1) + alpha_sf * fast_pred(t-1)) / K_slow;
    competition_slow = std::min(Type(1.0), competition_slow);
    
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - competition_slow) * temp_effect;
    Type slow_predation = std::min(slow_pred(t-1), F_slow * cots_pred(t-1));
    
    slow_pred(t) = std::max(eps, slow_pred(t-1) + slow_growth - slow_predation);
  }
  
  // Calculate negative log-likelihood
  Type min_sigma = Type(0.01);
  
  for (int t = 0; t < n_steps; t++) {
    // COTS abundance likelihood
    Type sigma_cots_t = std::max(min_sigma, sigma_cots);
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_t, true);
    
    // Slow-growing coral cover likelihood
    Type sigma_slow_t = std::max(min_sigma, sigma_slow);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_t, true);
    
    // Fast-growing coral cover likelihood
    Type sigma_fast_t = std::max(min_sigma, sigma_fast);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_t, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
