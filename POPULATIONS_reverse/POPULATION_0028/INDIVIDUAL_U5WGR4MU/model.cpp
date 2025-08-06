#include <TMB.hpp>
using namespace Rcpp;

// Data vectors for observations
vector<double> COTS_dat;    // Adult COTS abundance (individuals/m2)
vector<double> slow_dat;    // Slow-growing coral cover (%)
vector<double> fast_dat;    // Fast-growing coral cover (%)
vector<int> Year;           // Observation years

// Parameters
double cots_recruitment;     // Recruitment rate from larvae (year^-1)
double cots_mortality;       // Density-dependent mortality rate (-)
double slow_coral_growth;    // Slow-growing coral annual growth rate (% year^-1)
double fast_coral_growth;    // Fast-growing coral annual growth rate (% year^-1)
double cots_pred_eff_slow;   // COTS predation efficiency on slow-growing corals (-)
double cots_pred_eff_fast;   // COTS predation efficiency on fast-growing corals (-) 
double temp_effect_recruit;  // Effect of SST on COTS recruitment (unitless)

// Initial conditions
vector<double> COTS_0(1, 0.5);    // Starting COTS abundance
vector<double> slow_0(1, 10.0);   // Initial slow-growing coral cover (%)
vector<double> fast_0(1, 12.77);  // Initial fast-growing coral cover (%)

// Function to calculate population processes
double calc_pop_process(const vector<double>& t, 
                        const vector<double>& COTS_prev,
                        const double& sst) {
    // Calculate recruitment based on temperature
    double recruitment = cots_recruitment * exp(temp_effect_recruit * (sst - 25));
    
    // Calculate mortality
    double mortality = cots_mortality * COTS_prev;
    
    // Calculate new abundance
    double new_COTS = COTS_prev + recruitment - mortality;
    
    return new_COTS;
}

// Function to calculate coral growth processes
double calc_coral_growth(const vector<double>& t,
                        const double& coral_type,
                        const double& coral_prev) {
    // Calculate growth based on type (0=slow, 1=fast)
    if(coralType == 0) {  // Slow-growing corals
        return coral_prev * (1 + slow_coral_growth);
    } else {               // Fast-growing corals
        return coral_prev * (1 + fast_coral_growth);
    }
}

// Function to calculate predation effects
double calc_predation(const vector<double>& t,
                     const double& coral_type,
                     const double& coral_prev,
                     const double& COTS_prev) {
    // Calculate predation based on type and susceptibility
    double pred_eff = (coral_type == 0) ? cots_pred_eff_slow : cots_pred_eff_fast;
    
    // Use a Michaelis-Menten-type functional response
    double max_pred_rate = 0.1;  // Maximum predation rate (unitless)
    double k_coralType = (coral_type == 0) ? 500 : 200;  // Saturation constants
    
    return coral_prev * (max_pred_rate * pred_eff * COTS_prev) / (k_coralType + COTS_prev);
}

// Main model function
template<typename T>
T objective_function() {
    TMB_OBJECTIVE();
    
    // Data declarations
    DATA_VECTOR(COTS_dat);   
    DATA_VECTOR(slow_dat);    
    DATA_VECTOR(fast_dat);    
    DATA_VECTOR(Year);        
    PARAMETER(cots_recruitment);
    PARAMETER(cots_mortality);
    PARAMETER(slow_coral_growth);
    PARAMETER(fast_coral_growth);
    PARAMETER(cots_pred_eff_slow);
    PARAMETER(cots_pred_eff_fast);
    PARAMETER(temp_effect_recruit);

    // Initialize variables
    vector<double> COTS(COTS_0.size(), 0.0);
    vector<double> slow(slow_0.size(), 0.0);
    vector<double> fast(fast_0.size(), 0.0);
    
    // Set initial conditions
    for(int i=0; i<COTS_0.size(); ++i) {
        COTS[i] = COTS_0[i];
        slow[i] = slow_0[i]; 
        fast[i] = fast_0[i];
    }

    // Process model calculations
    for(int t=1; t<Year.size(); ++t) {
        // Get current year and SST (would need to link to actual data)
        double current_year = Year[t];
        
        // Calculate new COTS abundance
        double sst = 25 + sin(current_year / 365);  // Simplified SST fluctuation
        COTS[t] = calc_pop_process(t, COTS[t-1], sst);
        
        // Calculate coral growth
        slow[t] = calc_coral_growth(t, 0, slow[t-1]);
        fast[t] = calc_coral_growth(t, 1, fast[t-1]);
        
        // Apply predation effects
        double slow_pred = calc_predation(t, 0, slow[t], COTS[t]);
        double fast_pred = calc_predation(t, 1, fast[t], COTS[t]);
        
        // Subtract predation from coral growth
        slow[t] -= slow_pred;
        fast[t] -= fast_pred;
        
        // Ensure no negative values
        if(slow[t] < 0) slow[t] = 1e-8;
        if(fast[t] < 0) fast[t] = 1e-8;
    }

    // Calculate log-likelihood with observations
    double log_lik = 0.0;
    
    for(int t=0; t<Year.size(); ++t) {
        // Use lognormal distribution for counts
        double mu_cots = COTS[t];
        double mu_slow = slow[t];
        double mu_fast = fast[t];
        
        // Observation model with log-link function
        log_lik += -0.5 * log(2 * pi() + 1e-8) 
                  - log(mu_cots + 1e-8) 
                  - (log(COTS_dat[t] + 1e-8) - log(mu_cots + 1e-8)) / (1e-4 + 1e-8);
                  
        log_lik += -0.5 * log(2 * pi() + 1e-8) 
                  - log(mu_slow + 1e-8) 
                  - (log(slow_dat[t] + 1e-8) - log(mu_slow + 1e-8)) / (1e-4 + 1e-8);
                  
        log_lik += -0.5 * log(2 * pi() + 1e-8) 
                  - log(mu_fast + 1e-8) 
                  - (log(fast_dat[t] + 1e-8) - log(mu_fast + 1e-8)) / (1e-4 + 1e-8);
    }
    
    return log_lik;
}
