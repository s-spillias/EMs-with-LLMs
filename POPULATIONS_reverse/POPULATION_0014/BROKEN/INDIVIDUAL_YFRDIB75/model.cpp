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
  PARAMETER(interference_coef);       // Interference competition coefficient among COTS
  
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
    // 1. Calculate temperature effect on coral growth (Gaussian response curve)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_tol_safe = CppAD::CondExpLt(temp_tol, Type(0.1), Type(0.1), temp_tol);
    Type temp_effect = exp(-0.5 * pow(temp_diff / temp_tol_safe, 2));
    
    // 2. Calculate total coral cover (food availability for COTS)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    total_coral = CppAD::CondExpLt(total_coral, eps, eps, total_coral);
    
    // 3. Calculate density-dependent predator efficiency (decreases with COTS density)
    // This represents interference competition among predators
    Type cots_density = cots_pred(t-1);
    cots_density = CppAD::CondExpLt(cots_density, eps, eps, cots_density);
    
    // Ensure interference_coef is positive
    Type interference = CppAD::CondExpLt(interference_coef, Type(0), Type(0), interference_coef);
    
    // Calculate predator efficiency with a more stable formulation
    Type predator_efficiency = Type(1.0) / (Type(1.0) + interference * cots_density);
    predator_efficiency = CppAD::CondExpLt(predator_efficiency, Type(0.1), Type(0.1), predator_efficiency);
    predator_efficiency = CppAD::CondExpGt(predator_efficiency, Type(1.0), Type(1.0), predator_efficiency);
    
    // 4. Calculate functional responses for COTS feeding on corals (Type II with interference)
    Type fast_cover = fast_pred(t-1);
    fast_cover = CppAD::CondExpLt(fast_cover, eps, eps, fast_cover);
    
    Type slow_cover = slow_pred(t-1);
    slow_cover = CppAD::CondExpLt(slow_cover, eps, eps, slow_cover);
    
    // Ensure attack rates and handling times are positive
    Type a_fast_pos = CppAD::CondExpLt(a_fast, Type(0), Type(0), a_fast);
    Type a_slow_pos = CppAD::CondExpLt(a_slow, Type(0), Type(0), a_slow);
    Type h_fast_pos = CppAD::CondExpLt(h_fast, Type(0), Type(0), h_fast);
    Type h_slow_pos = CppAD::CondExpLt(h_slow, Type(0), Type(0), h_slow);
    
    // Calculate denominator for functional response with safeguards
    Type denom = Type(1.0) + a_fast_pos * h_fast_pos * fast_cover + a_slow_pos * h_slow_pos * slow_cover;
    denom = CppAD::CondExpLt(denom, Type(1.0), Type(1.0), denom);
    
    // Calculate feeding rates with interference
    Type F_fast = predator_efficiency * a_fast_pos * fast_cover / denom;
    Type F_slow = predator_efficiency * a_slow_pos * slow_cover / denom;
    
    // 5. Calculate food limitation effect on COTS (smooth transition at threshold)
    // Use a more stable sigmoid function
    Type sigmoid_input = Type(0.5) * (total_coral - coral_threshold);
    sigmoid_input = CppAD::CondExpGt(sigmoid_input, Type(10), Type(10), sigmoid_input);
    sigmoid_input = CppAD::CondExpLt(sigmoid_input, Type(-10), Type(-10), sigmoid_input);
    Type food_limitation = Type(0.1) + Type(0.8) / (Type(1.0) + exp(-sigmoid_input));
    
    // 6. COTS population dynamics with density dependence, mortality, and immigration
    // Ensure growth parameters are reasonable
    Type r_cots_pos = CppAD::CondExpLt(r_cots, Type(0), Type(0), r_cots);
    Type K_cots_pos = CppAD::CondExpLt(K_cots, Type(0.1), Type(0.1), K_cots);
    Type m_cots_pos = CppAD::CondExpLt(m_cots, Type(0), Type(0), m_cots);
    
    // Calculate population changes with safeguards
    Type logistic_term = Type(1.0) - cots_density / K_cots_pos;
    logistic_term = CppAD::CondExpLt(logistic_term, Type(-1), Type(-1), logistic_term);
    
    Type cots_growth = r_cots_pos * cots_density * logistic_term * food_limitation;
    cots_growth = CppAD::CondExpLt(cots_growth, -Type(0.9) * cots_density, -Type(0.9) * cots_density, cots_growth);
    
    Type cots_mortality = m_cots_pos * cots_density;
    cots_mortality = CppAD::CondExpGt(cots_mortality, Type(0.9) * cots_density, Type(0.9) * cots_density, cots_mortality);
    
    Type cots_immigration = imm_effect * cotsimm_dat(t-1);
    cots_immigration = CppAD::CondExpLt(cots_immigration, Type(0), Type(0), cots_immigration);
    
    // Update COTS population with bounds
    cots_pred(t) = cots_density + cots_growth - cots_mortality + cots_immigration;
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), eps, eps, cots_pred(t));
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(5.0), Type(5.0), cots_pred(t));
    
    // 7. Coral dynamics with competition and COTS predation
    // Ensure growth parameters are reasonable
    Type r_fast_pos = CppAD::CondExpLt(r_fast, Type(0), Type(0), r_fast);
    Type r_slow_pos = CppAD::CondExpLt(r_slow, Type(0), Type(0), r_slow);
    Type K_fast_pos = CppAD::CondExpLt(K_fast, Type(1.0), Type(1.0), K_fast);
    Type K_slow_pos = CppAD::CondExpLt(K_slow, Type(1.0), Type(1.0), K_slow);
    
    // Fast-growing coral
    // Use a more stable competition formulation
    Type fast_competition = Type(1.0) - (fast_cover + alpha_fs * slow_cover) / K_fast_pos;
    fast_competition = CppAD::CondExpLt(fast_competition, Type(-0.5), Type(-0.5), fast_competition);
    fast_competition = CppAD::CondExpGt(fast_competition, Type(1.0), Type(1.0), fast_competition);
    
    Type fast_growth = r_fast_pos * fast_cover * fast_competition * temp_effect;
    fast_growth = CppAD::CondExpLt(fast_growth, -Type(0.9) * fast_cover, -Type(0.9) * fast_cover, fast_growth);
    
    Type fast_predation = F_fast * cots_density;
    fast_predation = CppAD::CondExpGt(fast_predation, Type(0.9) * fast_cover, Type(0.9) * fast_cover, fast_predation);
    
    fast_pred(t) = fast_cover + fast_growth - fast_predation;
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), eps, eps, fast_pred(t));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), K_fast_pos, K_fast_pos, fast_pred(t));
    
    // Slow-growing coral
    Type slow_competition = Type(1.0) - (slow_cover + alpha_sf * fast_cover) / K_slow_pos;
    slow_competition = CppAD::CondExpLt(slow_competition, Type(-0.5), Type(-0.5), slow_competition);
    slow_competition = CppAD::CondExpGt(slow_competition, Type(1.0), Type(1.0), slow_competition);
    
    Type slow_growth = r_slow_pos * slow_cover * slow_competition * temp_effect;
    slow_growth = CppAD::CondExpLt(slow_growth, -Type(0.9) * slow_cover, -Type(0.9) * slow_cover, slow_growth);
    
    Type slow_predation = F_slow * cots_density;
    slow_predation = CppAD::CondExpGt(slow_predation, Type(0.9) * slow_cover, Type(0.9) * slow_cover, slow_predation);
    
    slow_pred(t) = slow_cover + slow_growth - slow_predation;
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), eps, eps, slow_pred(t));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), K_slow_pos, K_slow_pos, slow_pred(t));
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type min_sigma = Type(0.1); // Minimum standard deviation to prevent numerical issues
  
  for (int t = 0; t < n_steps; t++) {
    // COTS abundance likelihood
    Type sigma_cots_t = CppAD::CondExpLt(sigma_cots, min_sigma, min_sigma, sigma_cots);
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots_t, true);
    
    // Slow-growing coral cover likelihood
    Type sigma_slow_t = CppAD::CondExpLt(sigma_slow, min_sigma, min_sigma, sigma_slow);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow_t, true);
    
    // Fast-growing coral cover likelihood
    Type sigma_fast_t = CppAD::CondExpLt(sigma_fast, min_sigma, min_sigma, sigma_fast);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast_t, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
