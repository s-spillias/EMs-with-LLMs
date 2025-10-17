#include <TMB.hpp>

// Template for the COTS population dynamics model
template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA SECTION
  // ------------------------------------------------------------------------
  // These are the data inputs from the CSV files.
  
  DATA_VECTOR(Year);          // The years of the time series
  DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(sst_dat);       // Sea-surface temperature (not used in this version, but declared)
  
  int n_steps = Year.size();  // The number of time steps in the simulation
  
  // ------------------------------------------------------------------------
  // PARAMETER SECTION
  // ------------------------------------------------------------------------
  // These are the model parameters that will be estimated.
  
  // Coral growth parameters
  PARAMETER(log_r_fast);      // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(log_K_fast);      // Carrying capacity of fast-growing corals (%)
  PARAMETER(log_r_slow);      // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(log_K_slow);      // Carrying capacity of slow-growing corals (%)
  PARAMETER(log_alpha_slow);  // Competition coefficient of slow corals on fast corals
  PARAMETER(log_alpha_fast);  // Competition coefficient of fast corals on slow corals
  
  // COTS predation parameters (Holling Type II)
  PARAMETER(log_a_fast);      // Attack rate of COTS on fast-growing corals (m2/individual/year)
  PARAMETER(log_h_fast);      // Handling time of COTS on fast-growing corals (year/%)
  PARAMETER(log_a_slow);      // Attack rate of COTS on slow-growing corals (m2/individual/year)
  PARAMETER(log_h_slow);      // Handling time of COTS on slow-growing corals (year/%)
  
  // COTS life history parameters
  PARAMETER(log_e_fast);      // Assimilation efficiency of COTS from fast-growing corals
  PARAMETER(log_e_slow);      // Assimilation efficiency of COTS from slow-growing corals
  PARAMETER(log_m_cots);      // Natural mortality rate of COTS (year^-1)
  PARAMETER(log_q_cots);      // Density-dependent mortality coefficient for COTS (m2/individual/year)
  
  // Observation error parameters
  PARAMETER(log_sd_cots);     // Standard deviation for COTS abundance observations (log scale)
  PARAMETER(log_sd_fast);     // Standard deviation for fast coral cover observations (log scale)
  PARAMETER(log_sd_slow);     // Standard deviation for slow coral cover observations (log scale)
  
  // --- Parameter transformations ---
  // Parameters are estimated in log-space to ensure they are positive.
  Type r_fast = exp(log_r_fast);
  Type K_fast = exp(log_K_fast);
  Type r_slow = exp(log_r_slow);
  Type K_slow = exp(log_K_slow);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type a_fast = exp(log_a_fast);
  Type h_fast = exp(log_h_fast);
  Type a_slow = exp(log_a_slow);
  Type h_slow = exp(log_h_slow);
  Type e_fast = exp(log_e_fast);
  Type e_slow = exp(log_e_slow);
  Type m_cots = exp(log_m_cots);
  Type q_cots = exp(log_q_cots);
  Type sd_cots = exp(log_sd_cots);
  Type sd_fast = exp(log_sd_fast);
  Type sd_slow = exp(log_sd_slow);
  
  // ------------------------------------------------------------------------
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------
  // A short description of the model's core equations:
  // 1. Fast Coral Cover: Logistic growth minus COTS predation (Holling Type II).
  //    fast_pred(t) = fast_pred(t-1) + Growth - Predation
  // 2. Slow Coral Cover: Logistic growth minus COTS predation (Holling Type II).
  //    slow_pred(t) = slow_pred(t-1) + Growth - Predation
  // 3. COTS Abundance: Growth from predation minus natural and density-dependent mortality, plus immigration.
  //    cots_pred(t) = cots_pred(t-1) + Growth - Mortality + Immigration
  
  // --- Prediction vectors ---
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);
  
  // --- Initial conditions ---
  // Initialize the model predictions with the first data point.
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // --- Time loop for simulation ---
  for (int t = 1; t < n_steps; ++t) {
    // --- Intermediate terms for clarity ---
    
    // Holling Type II denominator: represents predator saturation from handling both prey types.
    Type holling_denom = Type(1.0) + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1) + Type(1e-8);
    
    // Predation on fast corals
    Type predation_on_fast = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / holling_denom;
    
    // Predation on slow corals
    Type predation_on_slow = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / holling_denom;
    
    // --- State variable predictions ---
    
    // 1. Fast-growing coral dynamics
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + alpha_slow * slow_pred(t-1)) / (K_fast + Type(1e-8)));
    fast_pred(t) = fast_pred(t-1) + fast_growth - predation_on_fast;
    
    // 2. Slow-growing coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - (slow_pred(t-1) + alpha_fast * fast_pred(t-1)) / (K_slow + Type(1e-8)));
    slow_pred(t) = slow_pred(t-1) + slow_growth - predation_on_slow;
    
    // 3. COTS dynamics
    Type cots_growth = e_fast * predation_on_fast + e_slow * predation_on_slow;
    Type cots_mortality = m_cots * cots_pred(t-1) + q_cots * cots_pred(t-1) * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mortality + cotsimm_dat(t-1);
    
    // --- Numerical stability constraints ---
    // Ensure predictions do not fall below a small positive number.
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.0), fast_pred(t), Type(1e-8));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.0), slow_pred(t), Type(1e-8));
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.0), cots_pred(t), Type(1e-8));
  }
  
  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  Type nll = 0.0; // Initialize negative log-likelihood
  
  // Loop over all time steps to compare predictions with observations.
  for (int t = 0; t < n_steps; ++t) {
    // Lognormal distribution is used for strictly positive data (abundances, cover).
    // This is equivalent to a normal distribution on the log-transformed data/predictions.
    // A small constant is added to prevent log(0).
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sd_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sd_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sd_slow, true);
  }
  
  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  // These variables will be available in the model output.
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  ADREPORT(r_fast);
  ADREPORT(K_fast);
  ADREPORT(r_slow);
  ADREPORT(K_slow);
  ADREPORT(a_fast);
  ADREPORT(h_fast);
  ADREPORT(a_slow);
  ADREPORT(h_slow);
  ADREPORT(e_fast);
  ADREPORT(e_slow);
  ADREPORT(m_cots);
  ADREPORT(q_cots);
  
  return nll;
}
