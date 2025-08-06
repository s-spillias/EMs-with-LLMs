// Template Model Builder (TMB) model for episodic COTS outbreaks on the Great Barrier Reef
#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator()() {
  
  // DATA INPUTS:
  DATA_VECTOR(Year);             // Years (integer values)
  DATA_VECTOR(cots_dat);             // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (Acropora, %)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (Faviidae/Porites, %)
  DATA_VECTOR(sst_dat);              // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration rate (individuals/m2/year)
  
  // PARAMETERS:
  PARAMETER(growth_rate);            // (year^-1) Intrinsic growth rate of COTS
  PARAMETER(coral_predation_eff_fast); // (unitless) Predation efficiency on fast-growing coral
  PARAMETER(coral_predation_eff_slow); // (unitless) Predation efficiency on slow-growing coral
  PARAMETER(outbreak_threshold);     // (normalized unit) Threshold triggering outbreak dynamics
  PARAMETER(recovery_rate);          // (year^-1) Coral recovery rate
  PARAMETER(log_std_cots);           // (log-scale) Log standard deviation for COTS observation error
  Type std_cots = exp(log_std_cots);
  
  int n = cots_dat.size();
  vector<Type> cots_pred(n), fast_pred(n), slow_pred(n);
  Type nll = 0.0;
  
  // Initialize predictions with the first observed value (avoid data leakage)
  cots_pred(0) = cots_dat(0);  // Initial COTS abundance
  fast_pred(0) = fast_dat(0);  // Initial fast-growing coral cover
  slow_pred(0) = slow_dat(0);  // Initial slow-growing coral cover
  
  for (int t = 1; t < n; t++) {
    // 1. COTS dynamics:
    // Calculate resource limitation (saturating function with small constant to avoid division by zero)
    Type resource_limitation = fast_pred(t-1) / (fast_pred(t-1) + Type(1e-8)); // (unitless)
    // Outbreak mechanism: If previous COTS exceed outbreak_threshold, enhanced growth factor applies
    Type outbreak_factor = (cots_pred(t-1) > outbreak_threshold ? Type(1.5) : Type(0.5));
    // Update COTS abundance using previous state, growth, resource limitation and immigration; note: using previous timestep to avoid leakage.
    cots_pred(t) = cots_pred(t-1) + growth_rate * cots_pred(t-1) * resource_limitation * outbreak_factor 
                   + cotsimm_dat(t-1); // (individuals/m2)
    // Environmental forcing: modify COTS growth by sea-surface temperature anomaly (relative to 26°C)
    cots_pred(t) += (sst_dat(t-1) - Type(26.0)) * Type(0.1);
    
    // 2. Fast-growing coral dynamics:
    // Coral loss by predation and recovery towards a presumed carrying capacity of 100 (% cover)
    fast_pred(t) = fast_pred(t-1) - coral_predation_eff_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(1e-8))
                   + recovery_rate * (Type(100) - fast_pred(t-1)) * Type(0.1);
    
    // 3. Slow-growing coral dynamics:
    slow_pred(t) = slow_pred(t-1) - coral_predation_eff_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(1e-8))
                   + recovery_rate * (Type(100) - slow_pred(t-1)) * Type(0.1);
    
    // 4. Likelihood Calculation:
    // Calculate negative log-likelihood with lognormal error distribution for strictly positive COTS data
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), std_cots, true);
  }
  
  // REPORT predictions for post-processing and diagnostics
  REPORT(cots_pred);  // Predicted COTS abundance (individuals/m2)
  REPORT(fast_pred);  // Predicted fast-growing coral cover (%)
  REPORT(slow_pred);  // Predicted slow-growing coral cover (%)
  
  // Equations Summary:
  // (1) COTS Dynamics: 
  //     cots_pred(t) = cots_pred(t-1) + growth_rate * cots_pred(t-1) * resource_limitation * outbreak_factor + cotsimm_dat(t-1)
  //                   + (sst_dat(t-1) - 26.0)*0.1
  // (2) Fast Coral Dynamics: 
  //     fast_pred(t) = fast_pred(t-1) - coral_predation_eff_fast * cots_pred(t-1) * fast_pred(t-1)/(fast_pred(t-1)+1e-8)
  //                      + recovery_rate * (100 - fast_pred(t-1))*0.1
  // (3) Slow Coral Dynamics: similar to fast coral dynamics with different predation efficiency.
  
  return nll;
}
