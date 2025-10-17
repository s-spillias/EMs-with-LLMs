#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
    // ------------------------------------------------------------------------
    // DATA
    // ------------------------------------------------------------------------
    // These are the observations provided to the model.
    DATA_VECTOR(Time);         // Time points of observations (days)
    DATA_VECTOR(N_dat);        // Observed nutrient concentration (g C m^-3)
    DATA_VECTOR(P_dat);        // Observed phytoplankton concentration (g C m^-3)
    DATA_VECTOR(Z_dat);        // Observed zooplankton concentration (g C m^-3)

    // ------------------------------------------------------------------------
    // PARAMETERS
    // ------------------------------------------------------------------------
    // These are the parameters we want to estimate.
    PARAMETER(V_max);      // Maximum phytoplankton uptake rate (day^-1)
    PARAMETER(K_N);        // Half-saturation constant for nutrient uptake (g C m^-3)
    PARAMETER(g_max);      // Maximum zooplankton grazing rate (day^-1)
    PARAMETER(K_P);        // Half-saturation constant for grazing (g C m^-3)
    PARAMETER(m_P);        // Phytoplankton quadratic mortality rate ((g C m^-3)^-1 day^-1)
    PARAMETER(m_Z);        // Zooplankton linear mortality rate (day^-1)
    PARAMETER(beta);       // Zooplankton assimilation efficiency (dimensionless)
    PARAMETER(epsilon);    // Zooplankton excretion rate (day^-1)
    PARAMETER(sigma_N);    // Log-normal standard deviation for Nutrient observations
    PARAMETER(sigma_P);    // Log-normal standard deviation for Phytoplankton observations
    PARAMETER(sigma_Z);    // Log-normal standard deviation for Zooplankton observations

    // ------------------------------------------------------------------------
    // MODEL SETUP
    // ------------------------------------------------------------------------
    int n_obs = Time.size(); // Number of observations

    // Create vectors to store model predictions
    vector<Type> N_pred(n_obs);
    vector<Type> P_pred(n_obs);
    vector<Type> Z_pred(n_obs);

    // Initialize predictions with the first data point
    N_pred(0) = N_dat(0);
    P_pred(0) = P_dat(0);
    Z_pred(0) = Z_dat(0);

    // Initialize negative log-likelihood
    Type nll = 0.0;

    // Small constant to prevent division by zero or log(0)
    Type epsilon_div = 1e-8;

    // ------------------------------------------------------------------------
    // PARAMETER BOUNDS & PENALTIES
    // ------------------------------------------------------------------------
    // Apply smooth quadratic penalties for parameters to stay within biological bounds.
    // CppAD::CondExpLt(var, bound, penalty, 0) provides a smooth "if(var < bound)" statement.
    // A large weight ensures the penalty is significant enough to constrain the parameter.
    Type penalty_weight = 1000.0;
    nll += penalty_weight * CppAD::CondExpLt(V_max, Type(0.0), pow(V_max - 0.0, 2), Type(0.0));   // V_max > 0
    nll += penalty_weight * CppAD::CondExpLt(K_N, Type(0.0), pow(K_N - 0.0, 2), Type(0.0));     // K_N > 0
    nll += penalty_weight * CppAD::CondExpLt(g_max, Type(0.0), pow(g_max - 0.0, 2), Type(0.0));   // g_max > 0
    nll += penalty_weight * CppAD::CondExpLt(K_P, Type(0.0), pow(K_P - 0.0, 2), Type(0.0));     // K_P > 0
    nll += penalty_weight * CppAD::CondExpLt(m_P, Type(0.0), pow(m_P - 0.0, 2), Type(0.0));     // m_P > 0
    nll += penalty_weight * CppAD::CondExpLt(m_Z, Type(0.0), pow(m_Z - 0.0, 2), Type(0.0));     // m_Z > 0
    nll += penalty_weight * CppAD::CondExpLt(beta, Type(0.0), pow(beta - 0.0, 2), Type(0.0));    // beta > 0
    nll += penalty_weight * CppAD::CondExpGt(beta, Type(1.0), pow(beta - 1.0, 2), Type(0.0));    // beta < 1
    nll += penalty_weight * CppAD::CondExpLt(epsilon, Type(0.0), pow(epsilon - 0.0, 2), Type(0.0)); // epsilon > 0
    nll += penalty_weight * CppAD::CondExpLt(sigma_N, Type(0.0), pow(sigma_N - 0.0, 2), Type(0.0)); // sigma_N > 0
    nll += penalty_weight * CppAD::CondExpLt(sigma_P, Type(0.0), pow(sigma_P - 0.0, 2), Type(0.0)); // sigma_P > 0
    nll += penalty_weight * CppAD::CondExpLt(sigma_Z, Type(0.0), pow(sigma_Z - 0.0, 2), Type(0.0)); // sigma_Z > 0

    // ------------------------------------------------------------------------
    // ECOLOGICAL PROCESSES & EQUATIONS
    // ------------------------------------------------------------------------
    // This model uses a system of ordinary differential equations (ODEs) to describe
    // the dynamics of Nutrients (N), Phytoplankton (P), and Zooplankton (Z).
    // The ODEs are solved numerically using the forward Euler method.
    //
    // 1. dN/dt = -Uptake + P_Mortality_Recycling + Z_Mortality_Recycling + Unassimilated_Grazing + Z_Excretion
    // 2. dP/dt = Uptake - Grazing - P_Mortality
    // 3. dZ/dt = Assimilated_Grazing - Z_Mortality - Z_Excretion
    //
    // Key terms:
    // - Uptake (by P): Michaelis-Menten kinetics on N. V_max * [N / (K_N + N)] * P
    // - Grazing (by Z): Holling Type II functional response on P. g_max * [P / (K_P + P)] * Z
    // - P_Mortality: Quadratic mortality for P. m_P * P^2
    // - Z_Mortality: Linear mortality for Z. m_Z * Z
    // - Assimilation: A fraction 'beta' of grazed P contributes to Z growth.
    // - Recycling: Unassimilated grazing, mortality, and excretion return to the N pool.

    for (int i = 1; i < n_obs; ++i) {
        // Time step (dt)
        Type dt = Time(i) - Time(i-1);

        // Previous state variables (from the model's prediction, not data)
        Type N_prev = N_pred(i-1);
        Type P_prev = P_pred(i-1);
        Type Z_prev = Z_pred(i-1);

        // 1. Phytoplankton nutrient uptake (Michaelis-Menten)
        Type p_uptake = V_max * (N_prev / (K_N + N_prev + epsilon_div)) * P_prev;

        // 2. Zooplankton grazing on phytoplankton (Holling Type II)
        Type z_grazing = g_max * (P_prev / (K_P + P_prev + epsilon_div)) * Z_prev;

        // 3. Phytoplankton mortality (quadratic)
        Type p_mortality = m_P * P_prev * P_prev;

        // 4. Zooplankton mortality (linear)
        Type z_mortality = m_Z * Z_prev;

        // 5. Zooplankton metabolic excretion
        Type z_excretion = epsilon * Z_prev;

        // 6. Zooplankton assimilated grazing (growth)
        Type z_assimilation = beta * z_grazing;
        
        // 7. Unassimilated grazing (sloppy eating -> nutrient recycling)
        Type unassimilated_grazing = (1.0 - beta) * z_grazing;

        // Calculate the derivatives (dN/dt, dP/dt, dZ/dt)
        Type dN = -p_uptake + p_mortality + z_mortality + unassimilated_grazing + z_excretion;
        Type dP = p_uptake - z_grazing - p_mortality;
        Type dZ = z_assimilation - z_mortality - z_excretion;

        // Update predictions using forward Euler method
        // Ensure predictions do not fall below a small positive number to maintain stability
        N_pred(i) = N_prev + dN * dt;
        if (N_pred(i) < epsilon_div) N_pred(i) = epsilon_div;

        P_pred(i) = P_prev + dP * dt;
        if (P_pred(i) < epsilon_div) P_pred(i) = epsilon_div;

        Z_pred(i) = Z_prev + dZ * dt;
        if (Z_pred(i) < epsilon_div) Z_pred(i) = epsilon_div;
    }

    // ------------------------------------------------------------------------
    // LIKELIHOOD CALCULATION
    // ------------------------------------------------------------------------
    // Calculate the negative log-likelihood of the data given the model predictions.
    // We assume a log-normal error distribution for the concentrations, which are
    // strictly positive quantities.
    
    // Add a small constant to sigma to prevent it from being zero, ensuring numerical stability.
    Type min_sigma = 1e-4;
    Type eff_sigma_N = sigma_N + min_sigma;
    Type eff_sigma_P = sigma_P + min_sigma;
    Type eff_sigma_Z = sigma_Z + min_sigma;

    for (int i = 0; i < n_obs; ++i) {
        // Add small constant to predictions and data to avoid log(0)
        Type N_pred_safe = N_pred(i) + epsilon_div;
        Type P_pred_safe = P_pred(i) + epsilon_div;
        Type Z_pred_safe = Z_pred(i) + epsilon_div;

        Type N_dat_safe = N_dat(i) + epsilon_div;
        Type P_dat_safe = P_dat(i) + epsilon_div;
        Type Z_dat_safe = Z_dat(i) + epsilon_div;

        // Calculate log-likelihood using normal distribution on log-transformed data.
        // This is equivalent to a log-normal distribution.
        // The formulation is: dnorm(log(data), mean_of_log_data, sd_of_log_data, log=true) - log(data)
        // where mean_of_log_data = log(predicted_mean) - sd^2 / 2
        nll -= (dnorm(log(N_dat_safe), log(N_pred_safe) - pow(eff_sigma_N, 2) / 2.0, eff_sigma_N, true) - log(N_dat_safe));
        nll -= (dnorm(log(P_dat_safe), log(P_pred_safe) - pow(eff_sigma_P, 2) / 2.0, eff_sigma_P, true) - log(P_dat_safe));
        nll -= (dnorm(log(Z_dat_safe), log(Z_pred_safe) - pow(eff_sigma_Z, 2) / 2.0, eff_sigma_Z, true) - log(Z_dat_safe));
    }

    // ------------------------------------------------------------------------
    // REPORTING
    // ------------------------------------------------------------------------
    // Report the predicted values for plotting and analysis.
    REPORT(N_pred);
    REPORT(P_pred);
    REPORT(Z_pred);

    return nll;
}
