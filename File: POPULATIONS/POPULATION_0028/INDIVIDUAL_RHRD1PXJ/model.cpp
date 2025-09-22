#include <TMB.hpp>

// Template Model Builder model for a plankton population dynamics model.
//
// Equations and comments:
// 1. Nutrient dynamics:
//    N_{t+1} = N_t + dt * ( -uptake*P + mortality_P * P + mortality_Z * Z )
// 2. Phytoplankton dynamics:
//    P_{t+1} = P_t + dt * ( efficiency_PP * uptake * N - grazing*P - mortality_P * P )
// 3. Zooplankton dynamics:
//    Z_{t+1} = Z_t + dt * ( efficiency_ZP * grazing * P - mortality_Z * Z )
//
// Small constants (eps) are used for numerical stability. Likelihoods are built
// using a lognormal distribution for each observed variable.
template<class Type>
Type objective_function<Type>::operator() ()
{
   using namespace density;

   // DATA INPUTS:
   DATA_VECTOR(time);     // Time vector (days)
   DATA_VECTOR(N_dat);    // Observed nutrient concentrations (g C m^-3)
   DATA_VECTOR(P_dat);    // Observed phytoplankton concentrations (g C m^-3)
   DATA_VECTOR(Z_dat);    // Observed zooplankton concentrations (g C m^-3)

   // PARAMETERS FOR OBSERVATION ERRORS:
   PARAMETER(log_sd_N);   // Log measurement error for nutrient (log(g C m^-3))
   PARAMETER(log_sd_P);   // Log measurement error for phytoplankton (log(g C m^-3))
   PARAMETER(log_sd_Z);   // Log measurement error for zooplankton (log(g C m^-3))
   Type sd_N = exp(log_sd_N); // Standard deviation for nutrients
   Type sd_P = exp(log_sd_P); // Standard deviation for phytoplankton
   Type sd_Z = exp(log_sd_Z); // Standard deviation for zooplankton

   // ECOLOGICAL PROCESS PARAMETERS:
   PARAMETER(growth_rate_NP); // Nutrient uptake rate by phytoplankton (day^-1)
   PARAMETER(half_sat_N);     // Half-saturation constant for nutrient uptake (g C m^-3)
   PARAMETER(efficiency_PP);  // Growth efficiency of phytoplankton (dimensionless; 0-1)
   PARAMETER(grazing_rate);   // Maximum grazing rate by zooplankton (day^-1)
   PARAMETER(half_sat_P);     // Half-saturation constant for grazing (g C m^-3)
   PARAMETER(efficiency_ZP);  // Assimilation efficiency for zooplankton (dimensionless; 0-1)
   PARAMETER(mortality_P);    // Phytoplankton mortality rate (day^-1)
   PARAMETER(mortality_Z);    // Zooplankton mortality rate (day^-1)

   // Numerical stability constant:
   Type eps = 1e-8;

   // Initialize Negative Log Likelihood:
   Type nll = 0.0;

   // Number of time steps:
   int n = time.size();

   // Vectors for model predictions:
   vector<Type> N_pred(n); // Predicted nutrient concentrations (g C m^-3)
   vector<Type> P_pred(n); // Predicted phytoplankton concentrations (g C m^-3)
   vector<Type> Z_pred(n); // Predicted zooplankton concentrations (g C m^-3)

   // INITIAL CONDITIONS: set to the first observation (avoid data leakage)
   N_pred(0) = N_dat(0);
   P_pred(0) = P_dat(0);
   Z_pred(0) = Z_dat(0);

   // Loop over time steps (starting at t = 1 to avoid using current observations)
   for(int t = 1; t < n; t++){
      // Time interval (days)
      Type dt = time(t) - time(t-1);
      dt = (dt < eps ? eps : dt);

      // Ecological Process Calculations:
      // 1. Nutrient uptake by phytoplankton (saturating function)
      Type uptake = growth_rate_NP * N_pred(t-1) / (half_sat_N + N_pred(t-1) + eps); 
      // 2. Grazing by zooplankton (Type-II functional response)
      Type grazing = grazing_rate * P_pred(t-1) / (half_sat_P + P_pred(t-1) + eps);

      // Equation 1: Nutrient dynamics (regeneration from mortality adds nutrients)
      N_pred(t) = N_pred(t-1) + dt * ( -uptake * P_pred(t-1) + mortality_P * P_pred(t-1) + mortality_Z * Z_pred(t-1) );
      
      // Equation 2: Phytoplankton dynamics (growth via nutrient uptake and losses)
      P_pred(t) = P_pred(t-1) + dt * ( efficiency_PP * uptake * N_pred(t-1) - grazing * P_pred(t-1) - mortality_P * P_pred(t-1) );
      
      // Equation 3: Zooplankton dynamics (growth from grazing and losses via mortality)
      Z_pred(t) = Z_pred(t-1) + dt * ( efficiency_ZP * grazing * P_pred(t-1) - mortality_Z * Z_pred(t-1) );

      // Ensure predictions remain strictly positive (prevents division by zero, etc.)
      N_pred(t) = (N_pred(t) < eps ? eps : N_pred(t));
      P_pred(t) = (P_pred(t) < eps ? eps : P_pred(t));
      Z_pred(t) = (Z_pred(t) < eps ? eps : Z_pred(t));
      
      // LIKELIHOOD CALCULATION: using lognormal error (data are strictly positive)
      nll -= dlnorm(N_dat(t), log(N_pred(t)), max(sd_N, Type(1e-3)), true);
      nll -= dlnorm(P_dat(t), log(P_pred(t)), max(sd_P, Type(1e-3)), true);
      nll -= dlnorm(Z_dat(t), log(Z_pred(t)), max(sd_Z, Type(1e-3)), true);
   }

   // REPORT predicted time series for later analysis:
   REPORT(N_pred); // Report predicted nutrient concentrations
   REPORT(P_pred); // Report predicted phytoplankton concentrations
   REPORT(Z_pred); // Report predicted zooplankton concentrations

   return nll;
}
