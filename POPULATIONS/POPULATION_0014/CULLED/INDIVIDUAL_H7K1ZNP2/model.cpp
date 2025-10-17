#include <TMB.hpp>

// Function to ensure predictions remain positive, adding a penalty if they fall below a minimum threshold
template <class Type>
Type posfun(Type x, Type min_val, Type &pen)
{
  pen += CppAD::CondExpLe(x, min_val, Type(1.0) * pow(log(x / min_val), 2), Type(0.0));
  return CppAD::CondExpGe(x, min_val, x, min_val);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  DATA_VECTOR(Year);          // Vector of years for the time series
  DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (individuals/m2/year)
  // sst_dat is available but not used in this model version
  // DATA_VECTOR(sst_dat);    // Observed Sea-Surface Temperature (°C)

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------
  // COTS parameters
  PARAMETER(cots_g1);         // COTS assimilation efficiency from predation (dimensionless)
  PARAMETER(cots_m1);         // COTS natural mortality rate (year^-1)
  PARAMETER(cots_m2);         // COTS density-dependent mortality coefficient ((ind/m^2)^-1 * year^-1)
  PARAMETER(cots_p_fast);     // COTS predation (attack) rate on fast corals (m^2 * ind^-1 * year^-1)
  PARAMETER(cots_p_slow);     // COTS predation (attack) rate on slow corals (m^2 * ind^-1 * year^-1)
  PARAMETER(cots_h);          // COTS predation handling time on corals (% cover^-1)

  // Coral parameters
  PARAMETER(fast_r);          // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(fast_K);          // Carrying capacity of fast-growing corals (% cover)
  PARAMETER(slow_r);          // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(slow_K);          // Carrying capacity of slow-growing corals (% cover)
  PARAMETER(alpha_fs);        // Competition coefficient of slow corals on fast corals (dimensionless)
  PARAMETER(alpha_sf);        // Competition coefficient of fast corals on slow corals (dimensionless)

  // Observation error parameters
  PARAMETER(log_sd_cots);     // Log of the standard deviation for COTS abundance
  PARAMETER(log_sd_fast);     // Log of the standard deviation for fast coral cover
  PARAMETER(log_sd_slow);     // Log of the standard deviation for slow coral cover

  // ------------------------------------------------------------------------
  // MODEL SETUP
  // ------------------------------------------------------------------------
  int n_steps = Year.size(); // Number of time steps in the simulation
  Type pen = 0.0;            // Penalty for non-positive state variable predictions
  Type nll = 0.0;            // Initialize negative log-likelihood

  // Create vectors to store model predictions
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);

  // ------------------------------------------------------------------------
  // INITIAL CONDITIONS
  // ------------------------------------------------------------------------
  // Initialize predictions with the first observed data point.
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ------------------------------------------------------------------------
  // ECOLOGICAL DYNAMICS & EQUATIONS
  // ------------------------------------------------------------------------
  // This model uses a forward Euler method to integrate the differential equations over time.
  //
  // EQUATION DESCRIPTIONS:
  // 1. COTS Population (cots_pred):
  //    dCOTS/dt = Growth - NaturalMortality - DensityMortality + Immigration
  //    - Growth: Proportional to COTS population and coral consumed, with a Holling Type II functional response.
  //    - NaturalMortality: A constant background mortality rate (m1).
  //    - DensityMortality: Mortality that increases with the square of COTS density (m2), representing self-limitation.
  //    - Immigration: External influx of COTS larvae (cotsimm_dat).
  //
  // 2. Fast-Growing Coral (fast_pred):
  //    dFAST/dt = LogisticGrowth - COTS_Predation
  //    - LogisticGrowth: Governed by intrinsic growth rate (r_fast) and carrying capacity (K_fast), reduced by competition from slow corals (alpha_fs).
  //    - COTS_Predation: Coral biomass lost to COTS predation, based on the Holling Type II response.
  //
  // 3. Slow-Growing Coral (slow_pred):
  //    dSLOW/dt = LogisticGrowth - COTS_Predation
  //    - LogisticGrowth: Governed by intrinsic growth rate (r_slow) and carrying capacity (K_slow), reduced by competition from fast corals (alpha_sf).
  //    - COTS_Predation: Coral biomass lost to COTS predation, based on the Holling Type II response.

  for (int i = 1; i < n_steps; ++i) {
    Type dt = Year(i) - Year(i-1); // Time step duration (should be 1 year)

    // Values from the previous time step (t-1)
    Type cots_t_minus_1 = cots_pred(i-1);
    Type fast_t_minus_1 = fast_pred(i-1);
    Type slow_t_minus_1 = slow_pred(i-1);

    // --- COTS Dynamics ---
    // Holling Type II functional response denominator
    Type functional_response_denom = Type(1.0) + cots_h * (fast_t_minus_1 + slow_t_minus_1) + Type(1e-8);
    // Total food available to COTS, weighted by predation rates
    Type food_eaten = (cots_p_fast * fast_t_minus_1 + cots_p_slow * slow_t_minus_1) / functional_response_denom;
    // COTS growth from consuming coral
    Type cots_growth = cots_g1 * food_eaten * cots_t_minus_1;
    // COTS natural mortality
    Type cots_natural_mortality = cots_m1 * cots_t_minus_1;
    // COTS density-dependent mortality
    Type cots_density_mortality = cots_m2 * cots_t_minus_1 * cots_t_minus_1;
    // Change in COTS population
    Type dCOTS = cots_growth - cots_natural_mortality - cots_density_mortality + cotsimm_dat(i-1);
    cots_pred(i) = cots_t_minus_1 + dCOTS * dt;

    // --- Fast Coral Dynamics ---
    // Logistic growth with competition from slow corals
    Type fast_growth = fast_r * fast_t_minus_1 * (Type(1.0) - (fast_t_minus_1 + alpha_fs * slow_t_minus_1) / (fast_K + Type(1e-8)));
    // Predation loss to COTS
    Type fast_predation_loss = (cots_p_fast * fast_t_minus_1 / functional_response_denom) * cots_t_minus_1;
    // Change in fast coral cover
    Type dFAST = fast_growth - fast_predation_loss;
    fast_pred(i) = fast_t_minus_1 + dFAST * dt;

    // --- Slow Coral Dynamics ---
    // Logistic growth with competition from fast corals
    Type slow_growth = slow_r * slow_t_minus_1 * (Type(1.0) - (slow_t_minus_1 + alpha_sf * fast_t_minus_1) / (slow_K + Type(1e-8)));
    // Predation loss to COTS
    Type slow_predation_loss = (cots_p_slow * slow_t_minus_1 / functional_response_denom) * cots_t_minus_1;
    // Change in slow coral cover
    Type dSLOW = slow_growth - slow_predation_loss;
    slow_pred(i) = slow_t_minus_1 + dSLOW * dt;

    // --- Numerical Stability ---
    // Ensure state variables remain positive using a penalty function
    cots_pred(i) = posfun(cots_pred(i), Type(1e-8), pen);
    fast_pred(i) = posfun(fast_pred(i), Type(1e-8), pen);
    slow_pred(i) = posfun(slow_pred(i), Type(1e-8), pen);
  }
  nll += pen; // Add penalty to the negative log-likelihood

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  // Calculate the negative log-likelihood of observing the data given the predictions.
  // A lognormal error distribution is assumed for all state variables, as they are strictly positive.
  // A small fixed minimum SD is added for numerical stability.
  Type min_sd = Type(0.01);

  Type sd_cots = exp(log_sd_cots) + min_sd;
  Type sd_fast = exp(log_sd_fast) + min_sd;
  Type sd_slow = exp(log_sd_slow) + min_sd;

  // Add a small constant to avoid log(0) for both data and predictions.
  // The results of log() are explicitly stored in intermediate vectors.
  // This forces the evaluation of the Eigen expression templates (e.g., from log(vector+scalar))
  // into concrete vector types that dnorm() can accept, resolving the compilation error.
  // Using .array() is good practice for element-wise operations.
  Type epsilon = 1e-8;

  vector<Type> log_cots_dat_vec = log(cots_dat.array() + epsilon);
  vector<Type> log_cots_pred_vec = log(cots_pred.array() + epsilon);
  nll -= dnorm(log_cots_dat_vec, log_cots_pred_vec, sd_cots, true).sum();

  vector<Type> log_fast_dat_vec = log(fast_dat.array() + epsilon);
  vector<Type> log_fast_pred_vec = log(fast_pred.array() + epsilon);
  nll -= dnorm(log_fast_dat_vec, log_fast_pred_vec, sd_fast, true).sum();

  vector<Type> log_slow_dat_vec = log(slow_dat.array() + epsilon);
  vector<Type> log_slow_pred_vec = log(slow_pred.array() + epsilon);
  nll -= dnorm(log_slow_dat_vec, log_slow_pred_vec, sd_slow, true).sum();

  // ------------------------------------------------------------------------
  // REPORTING
  // ------------------------------------------------------------------------
  // Report predicted time series for plotting and analysis
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Report estimated standard deviations for uncertainty assessment
  ADREPORT(sd_cots);
  ADREPORT(sd_fast);
  ADREPORT(sd_slow);

  return nll;
}
