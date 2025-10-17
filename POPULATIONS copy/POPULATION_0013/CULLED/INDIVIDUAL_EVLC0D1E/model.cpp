#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA
  // ------------------------------------------------------------------------
  
  // Time vector
  DATA_VECTOR(Year); // Time steps in years

  // Response variables
  DATA_VECTOR(cots_dat); // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%)

  // Forcing variables
  DATA_VECTOR(sst_dat); // Observed sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // Observed COTS larval immigration (individuals/m2/year)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------

  // Coral dynamics
  PARAMETER(log_r_F);      // log of intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(log_r_S);      // log of intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(log_K_coral);  // log of total coral carrying capacity (%)
  PARAMETER(log_m_F_sst);  // log of fast coral mortality rate due to high SST (degree^-1 year^-1)
  PARAMETER(log_m_S_sst);  // log of slow coral mortality rate due to high SST (degree^-1 year^-1)
  PARAMETER(T_bleach_F);   // Bleaching temperature threshold for fast corals (Celsius)
  PARAMETER(T_bleach_S);   // Bleaching temperature threshold for slow corals (Celsius)
  PARAMETER(log_k_bleach); // log of steepness of the logistic bleaching response

  // COTS dynamics
  PARAMETER(log_a_F);      // log of COTS attack rate on fast corals (m^2 ind^-1 year^-1)
  PARAMETER(log_a_S);      // log of COTS attack rate on slow corals (m^2 ind^-1 year^-1)
  PARAMETER(log_h);        // log of COTS handling time on corals (year)
  PARAMETER(log_e_C);      // log of COTS conversion efficiency from coral to COTS
  PARAMETER(log_m_pred);   // log of maximum COTS mortality rate from predation at low densities
  PARAMETER(log_m_C_dd);   // log of COTS density-dependent mortality coefficient (m^2 ind^-1 year^-1)
  PARAMETER(log_C_escape); // log of COTS density for predator satiation escape (ind/m^2)
  PARAMETER(log_C_allee);  // log of COTS density Allee threshold for reproduction (ind/m^2)

  // Observation error
  PARAMETER(log_sd_cots);  // log of standard deviation for COTS data (log scale)
  PARAMETER(log_sd_fast);  // log of standard deviation for fast coral data (log scale)
  PARAMETER(log_sd_slow);  // log of standard deviation for slow coral data (log scale)

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------

  // Unpack parameters
  Type r_F = exp(log_r_F);
  Type r_S = exp(log_r_S);
  Type K_coral = exp(log_K_coral);
  Type m_F_sst = exp(log_m_F_sst);
  Type m_S_sst = exp(log_m_S_sst);
  Type k_bleach = exp(log_k_bleach);
  Type a_F = exp(log_a_F);
  Type a_S = exp(log_a_S);
  Type h = exp(log_h);
  Type e_C = exp(log_e_C);
  Type m_pred = exp(log_m_pred);
  Type m_C_dd = exp(log_m_C_dd);
  Type C_escape = exp(log_C_escape);
  Type C_allee = exp(log_C_allee);
  Type sd_cots = exp(log_sd_cots);
  Type sd_fast = exp(log_sd_fast);
  Type sd_slow = exp(log_sd_slow);

  // Number of time steps
  int n_t = Year.size();

  // Prediction vectors
  vector<Type> cots_pred(n_t);
  vector<Type> fast_pred(n_t);
  vector<Type> slow_pred(n_t);

  // Initialize predictions with the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Negative log-likelihood
  Type nll = 0.0;

  // ------------------------------------------------------------------------
  // EQUATION DESCRIPTIONS
  // ------------------------------------------------------------------------
  // 1. COTS Predation (Holling Type II): Predation rate on fast (P_F) and slow (P_S) corals.
  //    P_coral = (a_coral * Coral) / (1 + a_F * h * FastCoral + a_S * h * SlowCoral)
  // 2. Total Consumption: Total amount of each coral type consumed by the COTS population.
  //    Consumed_coral = P_coral * COTS
  // 3. Coral Bleaching: Temperature-dependent mortality using a logistic function.
  //    Bleach_Effect = m_sst_coral / (1 + exp(-k_bleach * (SST - T_bleach)))
  // 4. Fast Coral Dynamics: Logistic growth minus COTS predation and bleaching mortality.
  //    dF/dt = r_F*F*(1-(F+S)/K) - Consumed_F - Bleach_Effect_F*F
  // 5. Slow Coral Dynamics: Logistic growth minus COTS predation and bleaching mortality.
  //    dS/dt = r_S*S*(1-(F+S)/K) - Consumed_S - Bleach_Effect_S*S
  // 6. COTS Dynamics: Growth from consumption (with Allee effect) minus predation (with predator satiation), density-dependent mortality, and immigration.
  //    dC/dt = e_C*(Consumed_F+Consumed_S)*(C/(C_allee+C)) - (m_pred*C)/(1+(C/C_escape)^2) - m_C_dd*C^2 + Immigration
  // ------------------------------------------------------------------------

  // ------------------------------------------------------------------------
  // PROCESS MODEL
  // ------------------------------------------------------------------------
  for (int t = 1; t < n_t; ++t) {
    // Previous time step values (for readability)
    Type C_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);
    Type SST_curr = sst_dat(t);

    // Numerical stability constant
    Type epsilon = 1e-8;

    // 1. COTS Predation (Holling Type II functional response)
    Type predation_denominator = 1.0 + a_F * h * F_prev + a_S * h * S_prev;
    Type consumed_per_capita_F = (a_F * F_prev) / (predation_denominator + epsilon);
    Type consumed_per_capita_S = (a_S * S_prev) / (predation_denominator + epsilon);

    // 2. Total Consumption by COTS population
    Type total_consumption_F = consumed_per_capita_F * C_prev;
    Type total_consumption_S = consumed_per_capita_S * C_prev;

    // 3. Coral Bleaching mortality from SST
    Type bleach_effect_F = m_F_sst / (1.0 + exp(-k_bleach * (SST_curr - T_bleach_F)));
    Type bleach_effect_S = m_S_sst / (1.0 + exp(-k_bleach * (SST_curr - T_bleach_S)));
    Type bleaching_loss_F = bleach_effect_F * F_prev;
    Type bleaching_loss_S = bleach_effect_S * S_prev;

    // 4. Fast Coral Dynamics
    Type fast_growth = r_F * F_prev * (1.0 - (F_prev + S_prev) / (K_coral + epsilon));
    fast_pred(t) = F_prev + fast_growth - total_consumption_F - bleaching_loss_F;

    // 5. Slow Coral Dynamics
    Type slow_growth = r_S * S_prev * (1.0 - (F_prev + S_prev) / (K_coral + epsilon));
    slow_pred(t) = S_prev + slow_growth - total_consumption_S - bleaching_loss_S;

    // 6. COTS Dynamics
    Type allee_effect = C_prev / (C_allee + C_prev + epsilon);
    Type cots_growth = e_C * (total_consumption_F + total_consumption_S) * allee_effect;
    Type predation_mortality = (m_pred * C_prev) / (1.0 + pow(C_prev / (C_escape + epsilon), 2.0));
    Type dd_mortality = m_C_dd * C_prev * C_prev;
    Type cots_mortality = predation_mortality + dd_mortality;
    cots_pred(t) = C_prev + cots_growth - cots_mortality + cotsimm_dat(t);

    // Ensure predictions are non-negative
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0.0), cots_pred(t), Type(0.0));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0.0), fast_pred(t), Type(0.0));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0.0), slow_pred(t), Type(0.0));
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD
  // ------------------------------------------------------------------------
  
  // Add a small constant to prevent issues with log(0) and small SDs
  Type min_sd = 1e-4;
  Type log_epsilon = 1e-8;

  // Lognormal distribution for all observations
  // This is equivalent to a normal distribution on the log-transformed data
  for (int t = 0; t < n_t; ++t) {
    nll -= dnorm(log(cots_dat(t) + log_epsilon), log(cots_pred(t) + log_epsilon), sd_cots + min_sd, true);
    nll -= dnorm(log(fast_dat(t) + log_epsilon), log(fast_pred(t) + log_epsilon), sd_fast + min_sd, true);
    nll -= dnorm(log(slow_dat(t) + log_epsilon), log(slow_pred(t) + log_epsilon), sd_slow + min_sd, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Report transformed parameters for interpretation
  REPORT(r_F);
  REPORT(r_S);
  REPORT(K_coral);
  REPORT(m_F_sst);
  REPORT(m_S_sst);
  REPORT(k_bleach);
  REPORT(a_F);
  REPORT(a_S);
  REPORT(h);
  REPORT(e_C);
  REPORT(m_pred);
  REPORT(m_C_dd);
  REPORT(C_escape);
  REPORT(C_allee);
  REPORT(sd_cots);
  REPORT(sd_fast);
  REPORT(sd_slow);

  return nll;
}
