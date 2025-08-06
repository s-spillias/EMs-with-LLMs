#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // 1. Data and Parameters:
  // ------------------------------------------------------------------------

  // 1.1. Data:
  DATA_VECTOR(Year);                // Time variable (Year)
  DATA_VECTOR(sst_dat);             // Sea Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);         // COTS larval immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);            // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);            // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);            // Fast-growing coral cover (%)

  // 1.2. Parameters:
  PARAMETER(cots_imm_rate);         // Baseline COTS larval immigration rate (individuals/m2/year)
  PARAMETER(sst_effect);            // Effect of sea surface temperature on COTS larval survival (Celsius^-1)
  PARAMETER(cots_repro_rate);       // COTS maximum reproductive rate (eggs/individual/year)
  PARAMETER(cots_carrying_capacity); // Carrying capacity of COTS (individuals/m2)
  PARAMETER(acropora_pref);         // COTS preference for Acropora coral (proportion)
  PARAMETER(faviidae_pref);         // COTS preference for Faviidae/Porites coral (proportion)
  PARAMETER(cots_consumption_rate);  // COTS consumption rate (m2 coral/individual/year)
  PARAMETER(coral_recovery_rate_fast); // Recovery rate of fast-growing coral (Acropora) (% cover/year)
  PARAMETER(coral_recovery_rate_slow); // Recovery rate of slow-growing coral (Faviidae/Porites) (% cover/year)
  PARAMETER(cots_natural_mortality); // Natural mortality rate of COTS (year^-1)
  PARAMETER(coral_carrying_capacity_fast); // Carrying capacity of fast-growing coral (Acropora) (% cover)
  PARAMETER(coral_carrying_capacity_slow); // Carrying capacity of slow-growing coral (Faviidae/Porites) (% cover)

  // 1.3. Constants:
  Type very_small = Type(1e-8);    // Small constant to prevent division by zero

  // 1.4. Model dimensions
  int n = Year.size();

  // 1.5. Initial conditions (using first observation)
  Type cots = cots_dat(0);
  Type slow = 20.0; // Initial slow coral cover
  Type fast = 15.0; // Initial fast coral cover

  // ------------------------------------------------------------------------
  // 2. Objective function:
  // ------------------------------------------------------------------------

  Type nll = 0.0;                   // Negative log-likelihood

  // ------------------------------------------------------------------------
  // 3. Model dynamics:
  // ------------------------------------------------------------------------

  // 3.1. Containers for predictions
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);

  // 3.2. Loop over time
  for(int t = 1; t < n; t++) {

    // 3.3. Environmental forcing
    Type sst = sst_dat(t);
    Type cots_imm = cotsimm_dat(t);

    // 3.4. COTS larval survival (modified by SST)
    Type larval_survival = cots_imm_rate * CppAD::exp(sst_effect * sst) + cots_imm;

    // 3.5. Coral consumption rates
    Type acropora_consumed = cots * cots_consumption_rate * acropora_pref * (fast / (fast + slow + very_small + Type(1.0)));
    Type faviidae_consumed = cots * cots_consumption_rate * faviidae_pref * (slow / (fast + slow + very_small + Type(1.0)));

    // 3.6. Coral growth and mortality (with carrying capacity)
    Type d_fast = coral_recovery_rate_fast * (coral_carrying_capacity_fast - fast) - acropora_consumed;
    Type d_slow = coral_recovery_rate_slow * (coral_carrying_capacity_slow - slow) - faviidae_consumed;

    // 3.7. COTS population dynamics
    Type cots_recruitment = cots_repro_rate * cots * (Type(1.0) - cots / (cots_carrying_capacity + very_small));
    Type d_cots = larval_survival + cots_recruitment - cots_natural_mortality * cots;

    // 3.8. Update populations
    cots = cots + d_cots;
    fast = fast + d_fast;
    slow = slow + d_slow;

    // 3.9. Bound predictions to a reasonable range
    cots = CppAD::abs(cots);  // Ensure COTS is non-negative
    fast = CppAD::CondExpGt(CppAD::abs(fast), coral_carrying_capacity_fast, coral_carrying_capacity_fast, CppAD::abs(fast)); // Ensure fast coral is non-negative and within carrying capacity
    slow = CppAD::CondExpGt(CppAD::abs(slow), coral_carrying_capacity_slow, coral_carrying_capacity_slow, CppAD::abs(slow)); // Ensure slow coral is non-negative and within carrying capacity

    // 3.10. Store predictions
    cots_pred(t) = cots;
    slow_pred(t) = slow;
    fast_pred(t) = fast;

  }

  // ------------------------------------------------------------------------
  // 4. Likelihood calculation:
  // ------------------------------------------------------------------------

  // 4.1. Fixed standard deviations for observations
  Type sd_cots = Type(0.2);
  Type sd_slow = Type(2.0);
  Type sd_fast = Type(2.0);

  // 4.2. Loop through observations and calculate likelihood
  for(int t = 1; t < n; t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t), sd_cots, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sd_slow, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sd_fast, true);
  }

  // ------------------------------------------------------------------------
  // 5. Reporting:
  // ------------------------------------------------------------------------

  ADREPORT(cots_imm_rate);
  ADREPORT(sst_effect);
  ADREPORT(cots_repro_rate);
  ADREPORT(cots_carrying_capacity);
  ADREPORT(acropora_pref);
  ADREPORT(faviidae_pref);
  ADREPORT(cots_consumption_rate);
  ADREPORT(coral_recovery_rate_fast);
  ADREPORT(coral_recovery_rate_slow);
  ADREPORT(cots_natural_mortality);

  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);

  return nll;
}
