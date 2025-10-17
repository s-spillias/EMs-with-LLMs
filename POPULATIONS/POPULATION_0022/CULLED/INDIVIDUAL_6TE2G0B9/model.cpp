#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------ //
  // MODEL INPUTS
  // ------------------------------------------------------------------------ //

  // DATA VECTORS
  DATA_VECTOR(Year);          // Year of observation
  DATA_VECTOR(sst_dat);       // Observed sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // Observed COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);      // Observed adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)

  // PARAMETERS
  // Coral dynamics
  PARAMETER(fast_growth_rate);   // Intrinsic growth rate of fast-growing corals (year^-1)
  PARAMETER(slow_growth_rate);   // Intrinsic growth rate of slow-growing corals (year^-1)
  PARAMETER(K_fast);             // Carrying capacity of fast-growing corals (%)
  PARAMETER(K_slow);             // Carrying capacity of slow-growing corals (%)
  PARAMETER(comp_fs);            // Competitive effect of fast coral on slow coral (dimensionless)
  PARAMETER(comp_sf);            // Competitive effect of slow coral on fast coral (dimensionless)
  PARAMETER(sst_opt_fast);       // Optimal SST for fast coral growth (Celsius)
  PARAMETER(sst_width_fast);     // SST tolerance width for fast coral growth (Celsius)
  PARAMETER(sst_opt_slow);       // Optimal SST for slow coral growth (Celsius)
  PARAMETER(sst_width_slow);     // SST tolerance width for slow coral growth (Celsius)

  // COTS-Coral interaction (predation)
  PARAMETER(cots_attack_rate);   // COTS attack rate on corals (% coral / ind / year)
  PARAMETER(cots_handling_time); // COTS handling time per unit of coral (% coral)^-1 * year
  PARAMETER(pref_fast);          // COTS preference for fast-growing corals (dimensionless)
  
  // COTS dynamics
  PARAMETER(assim_eff);          // COTS assimilation efficiency (converts % coral consumed to COTS ind/m2)
  PARAMETER(cots_nat_mort);      // COTS natural mortality rate (year^-1)
  PARAMETER(cots_self_reg);      // COTS density-dependent self-regulation (m^2/individual/year)

  // Observation error
  PARAMETER(log_sd_cots);        // Log of the standard deviation for the COTS abundance observation error
  PARAMETER(log_sd_fast);        // Log of the standard deviation for the fast coral cover observation error
  PARAMETER(log_sd_slow);        // Log of the standard deviation for the slow coral cover observation error

  // ------------------------------------------------------------------------ //
  // MODEL EQUATIONS
  // ------------------------------------------------------------------------ //
  // 1. SST effect on coral growth (Gaussian function):
  //    sst_effect = exp(-0.5 * ((sst - sst_opt) / sst_width)^2)
  // 2. Coral growth (logistic with competition and SST effect):
  //    growth_fast = r_f * F * (1 - (F + comp_sf * S) / K_f) * sst_effect_f
  //    growth_slow = r_s * S * (1 - (S + comp_fs * F) / K_s) * sst_effect_s
  // 3. COTS predation (multi-species Holling Type II functional response):
  //    pred_on_fast = (a * pref_f * F) / (1 + a * h * (pref_f * F + pref_s * S))
  //    pred_on_slow = (a * pref_s * S) / (1 + a * h * (pref_f * F + pref_s * S))
  // 4. COTS population dynamics:
  //    growth = C * assim_eff * (pred_on_fast + pred_on_slow)
  //    mortality = C * m_nat + C^2 * m_dens
  // 5. State variable updates (Euler method, dt=1 year):
  //    F(t) = F(t-1) + growth_fast - pred_on_fast * C(t-1)
  //    S(t) = S(t-1) + growth_slow - pred_on_slow * C(t-1)
  //    C(t) = C(t-1) + growth - mortality + immigration
  // ------------------------------------------------------------------------ //

  // Initialize negative log-likelihood
  Type nll = 0.0;

  // Get number of time steps
  int n_steps = Year.size();

  // Create prediction vectors
  vector<Type> cots_pred(n_steps);
  vector<Type> fast_pred(n_steps);
  vector<Type> slow_pred(n_steps);

  // Set initial conditions from the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Fixed parameter for COTS preference for slow-growing coral
  Type pref_slow = 1.0; // Set as the baseline preference

  // Time loop for predictions and likelihood calculation
  for (int t = 1; t < n_steps; ++t) {
    // Use previous time step's values for calculations
    Type cots_prev = cots_pred(t-1);
    Type fast_prev = fast_pred(t-1);
    Type slow_prev = slow_pred(t-1);
    Type sst_prev = sst_dat(t-1);
    Type cotsimm_prev = cotsimm_dat(t-1);

    // 1. SST effect on coral growth
    Type sst_effect_fast = exp(Type(-0.5) * pow((sst_prev - sst_opt_fast) / sst_width_fast, 2)); // Effect on fast corals
    Type sst_effect_slow = exp(Type(-0.5) * pow((sst_prev - sst_opt_slow) / sst_width_slow, 2)); // Effect on slow corals

    // 2. Coral growth dynamics
    Type fast_growth = fast_growth_rate * fast_prev * (Type(1.0) - (fast_prev + comp_sf * slow_prev) / K_fast) * sst_effect_fast;
    Type slow_growth = slow_growth_rate * slow_prev * (Type(1.0) - (slow_prev + comp_fs * fast_prev) / K_slow) * sst_effect_slow;

    // 3. COTS predation on corals (multi-species Holling Type II)
    Type denominator = Type(1.0) + cots_attack_rate * cots_handling_time * (pref_fast * fast_prev + pref_slow * slow_prev); // Denominator for functional response
    Type predation_on_fast = (cots_attack_rate * pref_fast * fast_prev) / denominator; // Predation rate per COTS on fast corals
    Type predation_on_slow = (cots_attack_rate * pref_slow * slow_prev) / denominator; // Predation rate per COTS on slow corals
    Type fast_predation_loss = predation_on_fast * cots_prev; // Total loss of fast coral cover
    Type slow_predation_loss = predation_on_slow * cots_prev; // Total loss of slow coral cover

    // 4. COTS population dynamics
    Type cots_growth = cots_prev * assim_eff * (predation_on_fast + predation_on_slow); // Growth from assimilated coral
    Type cots_mortality = cots_prev * cots_nat_mort + pow(cots_prev, 2) * cots_self_reg; // Natural and density-dependent mortality

    // 5. Update state variables (Euler integration with dt=1 year)
    fast_pred(t) = fast_prev + fast_growth - fast_predation_loss;
    slow_pred(t) = slow_prev + slow_growth - slow_predation_loss;
    cots_pred(t) = cots_prev + cots_growth - cots_mortality + cotsimm_prev;

    // Ensure predictions are non-negative
    fast_pred(t) = (fast_pred(t) > Type(1e-8)) ? fast_pred(t) : Type(1e-8);
    slow_pred(t) = (slow_pred(t) > Type(1e-8)) ? slow_pred(t) : Type(1e-8);
    cots_pred(t) = (cots_pred(t) > Type(1e-8)) ? cots_pred(t) : Type(1e-8);

    // 6. Likelihood calculation (lognormal distribution)
    // Add a small constant to data and predictions to prevent log(0)
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t)), exp(log_sd_cots), true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t)), exp(log_sd_fast), true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t)), exp(log_sd_slow), true);
  }

  // ------------------------------------------------------------------------ //
  // REPORTING
  // ------------------------------------------------------------------------ //
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  ADREPORT(fast_growth_rate);
  ADREPORT(slow_growth_rate);
  ADREPORT(K_fast);
  ADREPORT(K_slow);
  ADREPORT(comp_fs);
  ADREPORT(comp_sf);
  ADREPORT(sst_opt_fast);
  ADREPORT(sst_width_fast);
  ADREPORT(sst_opt_slow);
  ADREPORT(sst_width_slow);
  ADREPORT(cots_attack_rate);
  ADREPORT(cots_handling_time);
  ADREPORT(pref_fast);
  ADREPORT(assim_eff);
  ADREPORT(cots_nat_mort);
  ADREPORT(cots_self_reg);
  ADREPORT(log_sd_cots);
  ADREPORT(log_sd_fast);
  ADREPORT(log_sd_slow);

  return nll;
}
