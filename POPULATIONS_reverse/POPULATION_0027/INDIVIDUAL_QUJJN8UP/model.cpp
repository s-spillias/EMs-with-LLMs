#include <Rcpp.h>
#include <TMB.hpp>
using namespace Rcpp;

// [[Rcpp::depends(TMB)]]

template<typename T>
class Model {
private:
    // Data vectors
    const std::vector<T>& Year;
    const std::vector<T>& fast_coral_dat;
    const std::vector<T>& slow_coral_dat;
    const std::vector<T>& cots_dat;
    const std::vector<T>& sst_dat;
    
    // Parameters
    T cots_grow_rate; 
    T cots_mort_rate;
    T fast_coral_grow_rate;
    T slow_coral_grow_rate;
    T cots_coral_eaten_fast;
    T cots_coral_eaten_slow;
    T cots_density_dep;
    T temp_effect_scale;

    // Prediction vectors
    std::vector<T> cots_pred;
    std::vector<T> fast_coral_pred;
    std::vector<T> slow_coral_pred;

public:
    // Constructor to initialize parameters and predictions
    Model(const std::vector<T>& Year,
          const std::vector<T>& fast_coral_dat,
          const std::vector<T>& slow_coral_dat,
          const std::vector<T>& cots_dat,
          const std::vector<T>& sst_dat,
          const Rcpp::DataFrame& params) {
        this->Year = Year;
        this->fast_coral_dat = fast_coral_dat;
        this->slow_coral_dat = slow_coral_dat;
        this->cots_dat = cots_dat;
        this->sst_dat = sst_dat;

        // Initialize parameters from dataframe
        cots_grow_rate = params["cots_grow_rate"];
        cots_mort_rate = params["cots_mort_rate"];
        fast_coral_grow_rate = params["fast_coral_grow_rate"];
        slow_coral_grow_rate = params["slow_coral_grow_rate"];
        cots_coral_eaten_fast = params["cots_coral_eaten_fast"];
        cots_coral_eaten_slow = params["cots_coral_eaten_slow"];
        cots_density_dep = params["cots_density_dep"];
        temp_effect_scale = params["temp_effect_scale"];

        // Initialize prediction vectors
        cots_pred.resize(Year.size());
        fast_coral_pred.resize(Year.size());
        slow_coral_pred.resize(Year.size());

        // Set initial conditions (first year)
        fast_coral_pred[0] = 5.0; // Starting fast coral cover
        slow_coral_pred[0] = 10.0; // Starting slow coral cover
        cots_pred[0] = 0.3; // Starting COTS density
    }

    // Core model function
    template<typename T>
    inline T log_lik() {
        // Time series loop
        for(int t=1; t<Year.size(); ++t) {
            // Coral growth
            fast_coral_pred[t] = fast_coral_pred[t-1] * exp(fast_coral_grow_rate);
            slow_coral_pred[t] = slow_coral_pred[t-1] * exp(slow_coral_grow_rate);

            // COTS feeding and population dynamics
            T total_coral = fast_coral_pred[t-1] + slow_coral_pred[t-1];
            
            // Calculate consumption
            T cots_food = (cots_pred[t-1] * cots_coral_eaten_fast * fast_coral_pred[t-1]) +
                          (cots_pred[t-1] * cots_coral_eaten_slow * slow_coral_pred[t-1]);
            
            // Population growth with density dependence
            T cots_growth = cots_grow_rate * (1 - cots_pred[t-1]/(total_coral * cots_density_dep));
            cots_pred[t] = cots_pred[t-1] + cots_growth - cots_mort_rate * cots_pred[t-1];
            
            // Temperature effect
            T temp_effect = 1 + (sst_dat[t] - 28) * temp_effect_scale;
            cots_pred[t] *= temp_effect;
        }

        // Return predictions for reporting
        REPORT("cots_pred", cots_pred);
        REPORT("fast_coral_pred", fast_coral_pred);
        REPORT("slow_coral_pred", slow_coral_pred);

        return 0.0; // Placeholder log-likelihood
    }
};

// Main TMB function
template<typename T>
T objective_function(Rcpp::DataFrame params, std::vector<T> const& Year,
                    std::vector<T> const& fast_coral_dat,
                    std::vector<T> const& slow_coral_dat,
                    std::vector<T> const& cots_dat,
                    std::vector<T> const& sst_dat) {
    Model<T> model(Year, fast_coral_dat, slow_coral_dat, cots_dat, sst_dat, params);
    return model.log_lik<T>();
}

// Expose to R
RCPP_EXPORT SEXP TMB_model(SEXP Year_sexp,
                           SEXP fast_coral_dat_sexp,
                           SEXP slow_coral_dat_sexp,
                           SEXP cots_dat_sexp,
                           SEXP sst_dat_sexp) {
    // Convert SEXP objects to vectors
    std::vector<double> Year = Rcpp::as<std::vector<double>>(Year_sexp);
    std::vector<double> fast_coral_dat = Rcpp::as<std::vector<double>>(fast_coral_dat_sexp);
    std::vector<double> slow_coral_dat = Rcpp::as<std::vector<double>>(slow_coral_dat_sexp);
    std::vector<double> cots_dat = Rcpp::as<std::vector<double>>(cots_dat_sexp);
    std::vector<double> sst_dat = Rcpp::as<std::vector<double>>(sst_dat_sexp);

    // Create parameter vector
    std::vector<T> params = {
        0.5, // cots_grow_rate
        0.3, // cots_mort_rate 
        0.8, // fast_coral_grow_rate
        0.2, // slow_coral_grow_rate
        0.6, // cots_coral_eaten_fast
        0.4, // cots_coral_eaten_slow
        0.8, // cots_density_dep
        0.2  // temp_effect_scale
    };

    // Call the objective function with parameters and data
    return Rcpp::wrap(objective_function<Rcpp::double_>(
        Rcpp::DataFrame::create<std::string>(
            {"cots_grow_rate", "cots_mort_rate", "fast_coral_grow_rate",
             "slow_coral_grow_rate", "cots_coral_eaten_fast", "cots_coral_eaten_slow",
             "cots_density_dep", "temp_effect_scale"},
            params
        ),
        Year,
        fast_coral_dat,
        slow_coral_dat,
        cots_dat,
        sst_dat
    ));
}
