#include <TMB.hpp>

// Crown-of-Thorns outbreak model on the Great Barrier Reef
// Implements coupled dynamics of starfish outbreaks and coral community response
// Predictions (_pred) correspond to data (_dat) provided in input

template<class Type>
Type objective_function<Type>::operator() ()
{
    // === 1. DATA INPUTS ===
    DATA_VECTOR(Year);             // Time in years
    DATA_VECTOR(cots_dat);         // Observed COTS density (ind/m2)
    DATA_VECTOR(fast_dat);         // Observed fast-growing coral cover (%)
    DATA_VECTOR(slow_dat);         // Observed slow-growing coral cover (%)
    DATA_VECTOR(sst_dat);          // Observed temperature (°C)
    DATA_VECTOR(cotsimm_dat);      // Observed larval immigration rate (ind/m2/year)

    // === 2. PARAMETERS ===
    PARAMETER(r_cots);             // Intrinsic growth rate of COTS (yr^-1)
    PARAMETER(m_cots);             // Background mortality rate of COTS (yr^-1)
    PARAMETER(alpha_fast);         // Attack rate of COTS on fast coral (% consumed per starfish per yr)
    PARAMETER(alpha_slow);         // Attack rate of COTS on slow coral (% consumed per starfish per yr)
    PARAMETER(r_fast);             // Growth rate of fast coral (% cover yr^-1)
    PARAMETER(r_slow);             // Growth rate of slow coral (% cover yr^-1)
    PARAMETER(K_coral);            // Max total coral cover capacity (%)
    PARAMETER(temp_opt);           // Optimal SST for COTS growth (°C)
    PARAMETER(temp_sens);          // Temperature sensitivity scaling parameter
    PARAMETER(sd_obs_cots);        // Observation error SD for COTS (lognormal)
    PARAMETER(sd_obs_coral);       // Observation error SD for coral (% cover, Gaussian)

    int n = Year.size();

    // === 3. STATE VARIABLES ===
    vector<Type> cots_pred(n);     // Predicted COTS density
    vector<Type> fast_pred(n);     // Predicted fast coral cover (%)
    vector<Type> slow_pred(n);     // Predicted slow coral cover (%)

    // === 4. MODEL DYNAMICS ===

    // Initial conditions set to observed first-year values
    cots_pred(0) = cots_dat(0);
    fast_pred(0) = fast_dat(0);
    slow_pred(0) = slow_dat(0);

    for(int t=1; t<n; t++){
        // Coral total cover from previous step
        Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1);

        // Environmental effect: multiplicative temperature response, Gaussian around optimal
        Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt)/temp_sens, 2));

        // ---- COTS dynamics ----
        // Boom-bust via logistic-like growth modulated by coral resources
        Type resource_lim = coral_total_prev / (K_coral + Type(1e-8)); // proportional to coral availability
        Type cots_growth = r_cots * cots_pred(t-1) * resource_lim * temp_effect;
        Type cots_mort = m_cots * cots_pred(t-1);
        // Immigration adds to COTS each year
        cots_pred(t) = cots_pred(t-1) + cots_growth - cots_mort + cotsimm_dat(t-1);
        if(cots_pred(t) < Type(1e-8)) cots_pred(t) = Type(1e-8); // prevent negative

        // ---- Coral dynamics ----
        // Fast coral growth reduced as total cover approaches capacity
        Type fast_growth = r_fast * fast_pred(t-1) * (1 - coral_total_prev/K_coral);
        Type fast_loss = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (1.0 + fast_pred(t-1));
        fast_pred(t) = fast_pred(t-1) + fast_growth - fast_loss;
        if(fast_pred(t) < Type(1e-8)) fast_pred(t) = Type(1e-8);

        // Slow coral growth similarly capacity-limited
        Type slow_growth = r_slow * slow_pred(t-1) * (1 - coral_total_prev/K_coral);
        Type slow_loss = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (1.0 + slow_pred(t-1));
        slow_pred(t) = slow_pred(t-1) + slow_growth - slow_loss;
        if(slow_pred(t) < Type(1e-8)) slow_pred(t) = Type(1e-8);
    }

    // === 5. LIKELIHOOD ===
    Type nll = 0.0;
    Type min_sd = Type(0.05);

    for(int t=0; t<n; t++){
        // Lognormal likelihood for strictly positive COTS
        nll -= dnorm(log(cots_dat(t) + 1e-8),
                     log(cots_pred(t) + 1e-8),
                     sd_obs_cots + min_sd,
                     true);

        // Gaussian likelihood for coral % cover
        nll -= dnorm(fast_dat(t),
                     fast_pred(t),
                     sd_obs_coral + min_sd,
                     true);
        nll -= dnorm(slow_dat(t),
                     slow_pred(t),
                     sd_obs_coral + min_sd,
                     true);
    }

    // === 6. REPORTING ===
    REPORT(cots_pred);
    REPORT(fast_pred);
    REPORT(slow_pred);

    return nll;
}
