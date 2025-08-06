#include <TMB.hpp>

// Template Model Builder model for episodic COTS outbreaks
// Equations and parameter details:
// 1. COTS dynamics: cots_pred[t] = cots_pred[t-1] + growth - predation + outbreak_trigger
//    - growth is simulated with a saturating Michaelis-Menten function.
//    - outbreak_trigger is an environmentally-modulated term with a smooth peak.
// 2. Fast coral dynamics: fast_pred[t] = fast_pred[t-1] + coral_regen_rate_fast*(100 - fast_pred[t-1]) - predation_on_fast
// 3. Slow coral dynamics: slow_pred[t] = slow_pred[t-1] + coral_regen_rate_slow*(100 - slow_pred[t-1]) - predation_on_slow
//    - A small constant (1e-8) is used to prevent division by zero.
//    - Likelihoods use lognormal errors comparing log-transformed predictions and observations.

template<class Type>
Type objective_function<Type>::operator() ()
{
    // READ DATA
    DATA_VECTOR(Year);    // Year (time index)
    if(Year.size() == 0){
        error("Year is missing or empty");
    }
    DATA_VECTOR(cots_dat);    // Observed COTS density (individuals/m2)
    DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%)
    DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%)

    // MODEL PARAMETERS
    // Log of COTS intrinsic growth rate (year^-1); transformed to ensure positivity.
    PARAMETER(log_cots_growth_rate);
    // Log of efficiency of COTS predation on slow-growing corals (Faviidae/Porites); transformed to ensure positivity.
    PARAMETER(log_cots_predation_eff_faviidae);
    // Efficiency of COTS predation on fast-growing Acropora
    PARAMETER(cots_predation_eff_acropora);
    // Regeneration rate of slow-growing corals (year^-1)
    PARAMETER(coral_regen_rate_slow);
    // Regeneration rate of fast-growing corals (year^-1)
    PARAMETER(coral_regen_rate_fast);
    // Environmental modifier for outbreak triggering (unitless)
    PARAMETER(environment_mod);
    Type cots_growth_rate_val = exp(log_cots_growth_rate);
    Type cots_predation_eff_faviidae_val = exp(log_cots_predation_eff_faviidae);

    Type nll = 0.0;   // negative log likelihood

    int n = Year.size();
    // Vectors to store model predictions with suffix _pred as in _dat observations.
    vector<Type> cots_pred(n);
    vector<Type> fast_pred(n);
    vector<Type> slow_pred(n);

    // Initialize predictions using the first observed values to avoid data leakage.
    cots_pred[0] = cots_dat[0];
    fast_pred[0] = fast_dat[0];
    slow_pred[0] = slow_dat[0];

    // Loop over time, using previous time step values only.
    for(int t = 1; t < n; t++){
        // (1) COTS dynamics:
        // a. Growth term with resource limitation based on coral cover.
        Type coral_resource = fast_pred[t-1] + slow_pred[t-1];
        Type growth = cots_growth_rate_val * cots_pred[t-1] * (coral_resource / (coral_resource + cots_pred[t-1] + Type(1e-8)));
        // b. Predation terms on coral communities.
        Type predation_on_fast = cots_pred[t-1] * cots_predation_eff_acropora / (fast_pred[t-1] + Type(1e-8));
        Type predation_on_slow = cots_pred[t-1] * cots_predation_eff_faviidae_val / (slow_pred[t-1] + Type(1e-8));
        // c. Outbreak trigger: an environmental pulse modeled as a smooth Gaussian peak.
        Type outbreak_trigger = environment_mod * exp(-pow((Type(t) - Type(n)/2) / (Type(n)/10 + Type(1e-8)), 2));

        // Update COTS state using previous time step values only.
        cots_pred[t] = cots_pred[t-1] + growth + outbreak_trigger - (predation_on_fast + predation_on_slow);

        // (2) Fast coral dynamics:
        fast_pred[t] = fast_pred[t-1] + coral_regen_rate_fast * (Type(100) - fast_pred[t-1]) - predation_on_fast;

        // (3) Slow coral dynamics:
        slow_pred[t] = slow_pred[t-1] + coral_regen_rate_slow * (Type(100) - slow_pred[t-1]) - predation_on_slow;

        // Ensure predictions remain non-negative using a smooth rectifier.
        cots_pred[t] = Type(0.5) * (cots_pred[t] + fabs(cots_pred[t]));
        fast_pred[t] = Type(0.5) * (fast_pred[t] + fabs(fast_pred[t]));
        slow_pred[t] = Type(0.5) * (slow_pred[t] + fabs(slow_pred[t]));

        // Likelihood: using lognormal error distribution
        // A fixed minimum SD of 0.1 is used to avoid numerical issues.
        nll -= dnorm(log(cots_dat[t] + Type(1e-8)), log(cots_pred[t] + Type(1e-8)), Type(0.1), true);
        nll -= dnorm(log(fast_dat[t] + Type(1e-8)), log(fast_pred[t] + Type(1e-8)), Type(0.1), true);
        nll -= dnorm(log(slow_dat[t] + Type(1e-8)), log(slow_pred[t] + Type(1e-8)), Type(0.1), true);
    }

    // Reporting model predictions (with '_pred' suffix to match observation names)
    REPORT(cots_pred);
    REPORT(fast_pred);
    REPORT(slow_pred);

    return nll;
}
