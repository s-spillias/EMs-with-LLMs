#include <TMB.hpp>
using namespace Rcpp;

// [[Rcpp::export]]
template<typename T>
DataFrame model_function(const NumericVector<T>& t, 
                         const DataFrame& data,
                         const std::map<std::string, std::vector<T>>& parameters) {
    // Define parameters
    double r_cots = parameters.at("r_cots")[0];
    double K_cots = parameters.at("K_cots")[0];
    
    double a_slow = parameters.at("a_slow")[0];
    double a_fast = parameters.at("a_fast")[0];
    
    double temp_effect = parameters.at("temp_effect")[0];
    
    // Initialize variables
    std::vector<T> cots(1, 0.5);  // Initial COTS abundance
    std::vector<T> slow_coral(1, 10); // Initial slow-growing coral cover
    std::vector<T> fast_coral(1, 10); // Initial fast-growing coral cover
    
    // Time series observations
    std::vector<T> sst = data["sst_dat"];
    
    // Create vectors for predictions
    std::vector<T> cots_pred;
    std::vector<T> slow_pred; 
    std::vector<T> fast_pred;
    
    // Simulation loop
    for (int i = 0; i < t.size(); ++i) {
        // Get current time step
        double time = t[i];
        
        // Get environmental conditions
        double sst_i = sst[i];
        
        // Calculate growth rates
        double cots_growth = r_cots * (1 + temp_effect*(sst_i - 25));
        
        // Calculate predation terms
        double pred_slow = a_slow * slow_coral[i] / (1 + slow_coral[i]);
        double pred_fast = a_fast * fast_coral[i] / (1 + fast_coral[i]);
        
        // Population dynamics
        cots[i+1] = cots[i] + cots_growth * cots[i] * (1 - cots[i]/K_cots) -
                    0.5*(pred_slow + pred_fast);
                    
        // Coral growth (simplified)
        slow_coral[i+1] = slow_coral[i] + (1 - a_slow*cots[i]) * 0.1;
        fast_coral[i+1] = fast_coral[i] + (1 - a_fast*cots[i]) * 0.2;
        
        // Record predictions
        cots_pred.push_back(cots[i+1]);
        slow_pred.push_back(slow_coral[i+1]);
        fast_pred.push_back(fast_coral[i+1]);
    }
    
    // Calculate likelihood
    double log_lik = 0;
    
    // Add lognormal likelihood for COTS data
    std::vector<T> cots_obs = data["cots_dat"];
    T sigma_cots = 0.2;  // Fixed SD for numerical stability
    for (int i=0; i<cots_obs.size(); ++i) {
        T mu = log(cots_pred[i]);
        log_lik += dlnorm(log(cots_obs[i]), mu, sigma_cots);
    }
    
    // Add lognormal likelihood for coral data
    std::vector<T> slow_obs = data["slow_dat"];
    std::vector<T> fast_obs = data["fast_dat"];
    T sigma_coral = 0.15;
    
    for (int i=0; i<slow_obs.size(); ++i) {
        T mu_slow = log(slow_pred[i]);
        log_lik += dlnorm(log(slow_obs[i]), mu_slow, sigma_coral);
        
        T mu_fast = log(fast_pred[i]);
        log_lik += dlnorm(log(fast_obs[i]), mu_fast, sigma_coral);
    }
    
    // Report predictions
    REPORT("cots_pred", cots_pred);
    REPORT("slow_pred", slow_pred); 
    REPORT("fast_pred", fast_pred);
    
    return DataFrame::create(
        _["log_lik"] = {log_lik},
        _["cots_dat"] = cots_pred,
        _["slow_dat"] = slow_pred,
        _["fast_dat"] = fast_pred
    );
}
