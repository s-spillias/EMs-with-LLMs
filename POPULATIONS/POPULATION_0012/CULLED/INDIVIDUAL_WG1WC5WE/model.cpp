#include <TMB.hpp> // TMB header: provides templated functions and macros for the model

// Model Overview:
// 1. COTS Dynamics: Logistic growth modified by a smooth outbreak-triggered decline.
// 2. Fast-growing Coral (Acropora spp.) Dynamics: Recovery towards full cover minus losses due to COTS predation.
// 3. Slow-growing Coral (Faviidae spp. and Porites spp.) Dynamics: Similar recovery with slower dynamics and losses by COTS.
// Each parameter is commented with its units, origin, and role in the ecological processes.

template<class Type>
Type objective_function<Type>::operator() () {
    // --- DATA INPUTS ---
    // time: vector of years (as provided in the first column of the CSV)
    DATA_VECTOR(time);                   // (years)
    DATA_VECTOR(cots_dat);               // Observed COTS abundance (individuals/m2)
    DATA_VECTOR(fast_dat);               // Observed fast-growing coral cover (%) for Acropora spp.
    DATA_VECTOR(slow_dat);               // Observed slow-growing coral cover (%) for Faviidae spp. & Porites spp.
    
    // --- PARAMETERS ---
    // growth_rate_cots: Intrinsic growth rate of COTS (year^-1)
    // decay_rate_cots: Decline rate of COTS post-outbreak (year^-1)
    // coral_predation_efficiency: Efficiency of COTS predation on coral communities (per individual/m2)
    // carrying_capacity: Ecosystem carrying capacity for COTS (individuals/m2)
    // observed_sd: Standard deviation for lognormal observation errors
    PARAMETER(growth_rate_cots);         // (year^-1), literature/expert opinion
    PARAMETER(decay_rate_cots);          // (year^-1), literature
    PARAMETER(coral_predation_efficiency); // (m2/individual), expert opinion
    PARAMETER(carrying_capacity);        // (individuals/m2), literature
    PARAMETER(observed_sd);              // (log-scale units), initial estimate
    PARAMETER(outbreak_sharpness);       // (unitless), governs the steepness of the outbreak trigger function
    PARAMETER(handling_time);             // (time units), handling time for saturating predation response (Holling Type II)
    PARAMETER(outbreak_threshold);       // (unitless), fraction of carrying capacity for outbreak trigger
    PARAMETER(outbreak_decline_exponent);       // (unitless), exponent for non-linear outbreak decline dynamics. Values > 1 intensify the decline during outbreak.
    PARAMETER(resource_limitation_strength);       // (unitless), scaling factor representing effect of coral availability on COTS growth
    PARAMETER(hill_coefficient);           // (unitless), Hill coefficient for non-linear resource limitation
    PARAMETER(coral_half_saturation);      // (proportion), coral availability level for half-maximal effect in the Hill function
    PARAMETER(environmental_modifier);
    PARAMETER(predation_scaler);
    PARAMETER(coral_recovery_modifier);
    PARAMETER(coral_recovery_inhibition);
    
    // --- NUMERICAL STABILITY ---
    Type eps = Type(1e-8); // small constant to avoid division by zero
    
    int n = cots_dat.size();
    // Vectors to hold predictions (suffix _pred corresponds to observation names)
    vector<Type> cots_pred(n);
    vector<Type> fast_pred(n);
    vector<Type> slow_pred(n);
    
    // --- INITIAL CONDITIONS ---
    cots_pred[0] = cots_dat[0];  // Use first observation as initial state
    fast_pred[0] = fast_dat[0];
    slow_pred[0] = slow_dat[0];
    
    // --- MODEL EQUATIONS (loop over time steps; t uses previous state only) ---
    for(int t = 1; t < n; t++){
        // Equation 1: COTS Dynamics
        // Incorporate resource limitation by scaling growth with available coral cover (sum of fast and slow predictions).
        Type coral_availability = (fast_pred[t-1] + slow_pred[t-1]) / Type(200);
        Type coral_limitation = (pow(coral_availability, hill_coefficient) / (pow(coral_half_saturation, hill_coefficient) + pow(coral_availability, hill_coefficient)));
        Type growth = growth_rate_cots * cots_pred[t-1] * (1 - cots_pred[t-1] / (carrying_capacity + eps)) * (Type(1) + resource_limitation_strength * (Type(2) * coral_limitation - Type(1)));
        Type outbreak_trigger = 1 / (Type(1) + exp(-outbreak_sharpness * environmental_modifier * (cots_pred[t-1] - outbreak_threshold * carrying_capacity)));
        Type decline = decay_rate_cots * pow(cots_pred[t-1], outbreak_decline_exponent) * outbreak_trigger;
        cots_pred[t] = cots_pred[t-1] + growth - decline; // Updated COTS population
        
        // Equation 2: Fast-growing Coral Dynamics (Acropora spp.)
        //  - Recovery: Proportional to the gap to maximum cover (assumed 100%) modified by inhibition from high COTS levels
        //  - Decline: Losses due to predation by COTS with saturation at low coral cover
        fast_pred[t] = fast_pred[t-1] + Type(0.1) * coral_recovery_modifier * (Type(100) - fast_pred[t-1]) * (Type(1) - coral_recovery_inhibition * cots_pred[t-1] / (carrying_capacity + eps))
                     - (cots_pred[t-1] * coral_predation_efficiency * fast_pred[t-1] *
                        (fast_pred[t-1] / (fast_pred[t-1] + predation_scaler))) / (Type(1) + handling_time * fast_pred[t-1]);
        
        // Equation 3: Slow-growing Coral Dynamics (Faviidae/Porites spp.)
        //  - Recovery: Slower than fast-growing coral, modified by inhibition from high COTS levels
        //  - Decline: Affected by COTS predation with saturation at low coral cover
        slow_pred[t] = slow_pred[t-1] + Type(0.05) * (Type(100) - slow_pred[t-1]) * (Type(1) - coral_recovery_inhibition * cots_pred[t-1] / (carrying_capacity + eps))
                     - (cots_pred[t-1] * coral_predation_efficiency * slow_pred[t-1] *
                        (slow_pred[t-1] / (slow_pred[t-1] + predation_scaler))) / (Type(1) + handling_time * slow_pred[t-1]);
    }
    
    // --- LIKELIHOOD CALCULATION ---
    // Use lognormal distributions (log-transformed data) for strictly positive observations.
    // A fixed small standard deviation is enforced via observed_sd.
    Type jnll = 0.0;
    for(int t = 0; t < n; t++){
        jnll -= dnorm(log(cots_dat[t] + eps), log(cots_pred[t] + eps), observed_sd, true);
        jnll -= dnorm(log(fast_dat[t] + eps), log(fast_pred[t] + eps), observed_sd, true);
        jnll -= dnorm(log(slow_dat[t] + eps), log(slow_pred[t] + eps), observed_sd, true);
    }
    
    // --- REPORTING ---
    // Report all predicted variables with the _pred suffix as required.
    REPORT(cots_pred);  // Predicted COTS
    REPORT(fast_pred);  // Predicted Fast-growing Coral Cover
    REPORT(slow_pred);  // Predicted Slow-growing Coral Cover
    
    return jnll;
}
