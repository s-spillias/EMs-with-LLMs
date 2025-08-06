#include <TMB.hpp>
#include <vector>

using namespace Rcpp;

// [[Rcpp::export]]
template<typename T>
T model(T& data) {
    // Parameter declarations (these should come from parameters.json)
    double k_slow = 0.1;   // Slow-growing coral growth rate (year^-1)
    double k_fast = 0.2;   // Fast-growing coral growth rate (year^-1) 
    double alpha_prey = 0.5; // Prey selectivity coefficient
    double C_max_slow = 50.0; // Slow coral carrying capacity (% cover)
    double C_max_fast = 40.0; // Fast coral carrying capacity (% cover)
    
    double r = 0.3;        // Intrinsic growth rate (year^-1)
    double K_cots = 200.0; // COTS carrying capacity (ind/m^2)
    double mu_temp = 0.05; // Temperature-dependent mortality coefficient

    // Data declarations
    DATA_VECTOR(sst_dat);     // Sea surface temperature data
    DATA_VECTOR(cotsimm_dat); // Larval immigration data
    DATA_VECTOR(slow_dat);    // Slow-growing coral cover data
    DATA_VECTOR(fast_dat);    // Fast-growing coral cover data 
    DATA_VECTOR(cots_dat);    // COTS abundance data

    // Initialize variables
    double prev_slow = 10.0;     // Initial slow coral cover (%)
    double prev_fast = 10.0;      // Initial fast coral cover (%) 
    double prev_cots = 0.5;       // Initial COTS density (ind/m^2)
    
    // Time series predictions
    vector<double> pred_slow;
    vector<double> pred_fast;
    vector<double> pred_cots;
    
    int n = sst_dat.size();
    
    for(int t=0; t<n; t++) {
        // Coral growth rates with density dependence
        double slow_growth = (k_slow * prev_slow) / (1 + prev_slow/C_max_slow);
        double fast_growth = (k_fast * prev_fast) / (1 + prev_fast/C_max_fast);
        
        // COTS predation with prey selectivity
        double predation_slow = (alpha_prey * prev_cots * prev_slow) / (1 + prev_cots/K_cots);
        double predation_fast = ((1 - alpha_prey) * prev_cots * prev_fast) / (1 + prev_cots/K_cots);
        
        // Temperature-dependent mortality
        double cots_mortality = mu_temp * sst_dat[t] * prev_cots;
        
        // Population updates
        double new_slow = prev_slow + slow_growth - predation_slow;
        double new_fast = prev_fast + fast_growth - predation_fast;
        double new_cots = prev_cots + cotsimm_dat[t] - cots_mortality - (r * prev_cots);
        
        // Store predictions
        pred_slow.push_back(new_slow);
        pred_fast.push_back(new_fast); 
        pred_cots.push_back(new_cots);
        
        // Ensure no negative values and update previous states for next iteration
        prev_slow = std::max(0.0, new_slow);
        prev_fast = std::max(0.0, new_fast);
        prev_cots = std::max(0.0, new_cots);
    }
    
    // Calculate log-likelihood with fixed minimum SD to avoid numerical issues
    double log_lik = 0.0;
    const double min_sd = 1e-4;
    
    for(int t=0; t<n; t++) {
        // Slow coral likelihood
        log_lik -= R::dnorm(slow_dat[t], pred_slow[t], min_sd, true);
        
        // Fast coral likelihood  
        log_lik -= R::dnorm(fast_dat[t], pred_fast[t], min_sd, true);
        
        // COTS likelihood
        log_lik -= R::dnorm(cots_dat[t], pred_cots[t], min_sd, true);
    }
    
    return -log_lik;
}
