// TMB model for COTS outbreak dynamics
// Based on: 
// - COTS population growth with temperature-dependent reproduction
// - Size-selective predation on coral communities
// - Density-dependent mortality mechanisms
// - Environmental feedback loops

// Parameters
PARAMETER(growth_rate);        // Intrinsic COTS growth rate (year^-1)
PARAMETER(predation_eff);      // Maximum predation efficiency (m²/year)
PARAMETER(saturation_const);   // Saturation constant for predation (individuals/m²)
PARAMETER(temp_opt);           // Optimal SST for COTS reproduction (°C)
PARAMETER(temp_steepness);     // Temperature response steepness (1/°C)
PARAMETER(mortality_rate);     // Baseline COTS mortality rate (year^-1)
PARAMETER(coral_growth_slow);  // Slow coral growth rate (%/year)
PARAMETER(coral_growth_fast);  // Fast coral growth rate (%/year)
PARAMETER(coral_competition);  // Competition coefficient between coral types
PARAMETER(min_sd);             // Minimum standard deviation for likelihood

// Data
DATA_VECTOR(Year);             // Time vector
DATA_VECTOR(sst_dat);          // Observed SST (°C)
DATA_VECTOR(cots_dat);         // Observed COTS abundance (individuals/m²)
DATA_VECTOR(slow_dat);         // Observed slow coral cover (%)
DATA_VECTOR(fast_dat);         // Observed fast coral cover (%)

// Predictions
PREDC(cots_pred);              // Predicted COTS abundance
PREDC(slow_pred);              // Predicted slow coral cover
PREDC(fast_pred);              // Predicted fast coral cover

// Model equations
// 1. COTS population dynamics
//    dC/dt = growth_rate * C * (1 - C/K) + temperature_effect * C - predation_loss - mortality
//    K = saturation_const / (1 + exp(-temp_steepness*(sst - temp_opt)))
//    temperature_effect = (sst - temp_opt)^2 / (1 + (sst - temp_opt)^2)
//    predation_loss = predation_eff * C * (slow + fast) / (saturation_const + slow + fast)
// 2. Coral dynamics
//    dS/dt = coral_growth_slow * (1 - S/(S_max + F_max)) * (1 - F/(S_max + F_max)) - predation_loss
//    dF/dt = coral_growth_fast * (1 - F/(S_max + F_max)) * (1 - S/(S_max + F_max)) - predation_loss
//    S_max = 100, F_max = 100 (maximum cover values)

// Main model function
void model() {
  // Initialize predictions
  cots_pred = cots_dat;
  slow_pred = slow_dat;
  fast_pred = fast_dat;
  
  // Time loop
  for (int t = 1; t < Year.size(); ++t) {
    // 1. Temperature-dependent COTS growth
    double temp_diff = sst_dat[t-1] - temp_opt;
    double temp_effect = (temp_diff * temp_diff) / (1 + temp_diff * temp_diff);
    
    // 2. Carrying capacity based on temperature
    double K = saturation_const / (1 + exp(-temp_steepness * temp_diff));
    
    // 3. COTS population dynamics
    double predation_loss = predation_eff * cots_dat[t-1] * (slow_dat[t-1] + fast_dat[t-1]) 
                          / (saturation_const + slow_dat[t-1] + fast_dat[t-1] + 1e-8);
    
    double dC = growth_rate * cots_dat[t-1] * (1 - cots_dat[t-1]/K) 
               + temp_effect * cots_dat[t-1] 
               - predation_loss 
               - mortality_rate * cots_dat[t-1];
    
    // 4. Coral dynamics with competition
    double S = slow_dat[t-1];
    double F = fast_dat[t-1];
    double total = S + F + 1e-8;
    
    double dS = coral_growth_slow * (1 - S/100) * (1 - F/100) - predation_loss * S / total;
    double dF = coral_growth_fast * (1 - F/100) * (1 - S/100) - predation_loss * F / total;
    
    // 5. Update predictions
    cots_pred[t] = cots_dat[t-1] + dC;
    slow_pred[t] = S + dS;
    fast_pred[t] = F + dF;
    
    // 6. Bounded predictions (smooth constraints)
    slow_pred[t] = 100 * (1 - exp(-slow_pred[t]/100)) + 1e-8;
    fast_pred[t] = 100 * (1 - exp(-fast_pred[t]/100)) + 1e-8;
  }
  
  // 7. Likelihood calculation
  // Log-transformed data for better numerical stability
  double total_ll = 0.0;
  for (int t = 0; t < Year.size(); ++t) {
    double log_cots = log(cots_dat[t] + 1e-8);
    double log_slow = log(slow_dat[t] + 1e-8);
    double log_fast = log(fast_dat[t] + 1e-8);
    
    // Log-normal likelihood with minimum SD
    double sd = max(min_sd, 0.1 * (cots_dat[t] + 1e-8));
    double ll_cots = -0.5 * pow((log_cots - log(cots_pred[t] + 1e-8))/sd, 2) - log(sd) - 0.5*log(2*M_PI);
    
    sd = max(min_sd, 0.1 * (slow_dat[t] + 1e-8));
    double ll_slow = -0.5 * pow((log_slow - log(slow_pred[t] + 1e-8))/sd, 2) - log(sd) - 0.5*log(2*M_PI);
    
    sd = max(min_sd, 0.1 * (fast_dat[t] + 1e-8));
    double ll_fast = -0.5 * pow((log_fast - log(fast_pred[t] + 1e-8))/sd, 2) - log(sd) - 0.5*log(2*M_PI);
    
    // Accumulate likelihood
    total_ll += ll_cots + ll_slow + ll_fast;
  }
  
  // Report total log-likelihood
  ADREPORT(total_ll)
}

// Reporting section
void report() {
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
}
