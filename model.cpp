#include <TMB.hpp> // TMB library for Template Model Builder

template<class Type>
Type objective_function<Type>::operator() ()
{
    // DATA (observations)
    DATA_VECTOR(cots_dat);      // Observed COTS abundance (individuals/m2)
    DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%) 
    DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%) 
    DATA_VECTOR(sst_dat);       // Observed sea-surface temperature (°C)
    DATA_VECTOR(cotsimm_dat);   // Observed COTS immigration rate (individuals/m2/year)
    
    // PARAMETERS (to be estimated)
    PARAMETER(growth_rate);         // (year^-1) Intrinsic growth rate of COTS as determined from literature
    PARAMETER(slow_consumption);    // (% impact per unit cover) Consumption rate on slow-growing corals (expert opinion)
    PARAMETER(fast_consumption);    // (% impact per unit cover) Consumption rate on fast-growing corals (expert opinion)
    PARAMETER(temp_effect);         // (per °C) Temperature effect coefficient on growth (literature)
    PARAMETER(immigration_effect);  // (individuals/m2/year) Immigration effect magnitude (initial estimate)
    PARAMETER(log_sd);              // Log standard deviation for the error term; must be exponentiated (initial estimate)
    
    // SE standard deviation ensuring a small constant to avoid zero
    Type sd = fmax(exp(log_sd), Type(1e-8)); // The error standard deviation for the lognormal likelihood
    
    // Initialize negative log likelihood
    Type nll = Type(0);
    
    int n = cots_dat.size();                  // Number of observations
    vector<Type> cots_pred(n);                // Predicted COTS abundance (individuals/m2)
    
    /* Equations:
       1. Predicted COTS abundance is modeled as:
          cots_pred = growth_rate * observed_COTS 
                      - slow_consumption * slow_adjusted * observed_COTS 
                      - fast_consumption * fast_adjusted * observed_COTS 
                      + temp_effect * sst_dat 
                      + immigration_effect * cotsimm_dat
       2. slow_adjusted and fast_adjusted provide a smooth adjustment using a small constant to avoid division by zero.
       3. The likelihood is computed using a lognormal distribution ensuring all predictions are positive.
    */
    for(int i = 0; i < n; i++){
        Type slow_adj = slow_dat(i) + Type(1e-8);  // Adjust slow-growing coral cover (%)
        Type fast_adj = fast_dat(i) + Type(1e-8);  // Adjust fast-growing coral cover (%)
        
        // Equation 1: Compute predicted COTS abundance (individuals/m2)
        cots_pred(i) = growth_rate * cots_dat(i)                       // intrinsic growth (year^-1)
                        - slow_consumption * slow_adj * cots_dat(i)      // consumption effect on slow-growing corals (% impact)
                        - fast_consumption * fast_adj * cots_dat(i)      // consumption effect on fast-growing corals (% impact)
                        + temp_effect * sst_dat(i)                       // temperature influence (per °C)
                        + immigration_effect * cotsimm_dat(i);           // immigration contribution (individuals/m2/year)
        
        // Accumulate the negative log likelihood using a lognormal error distribution.
        Type pred = fmax(cots_pred(i), Type(1e-8));  // Ensure predicted abundance is positive
        nll -= dnorm(log(cots_dat(i) + Type(1e-8)), log(pred), sd, true);
    }
    
    // Reporting important parameters and predictions with the _pred suffix for predictions
    ADREPORT(growth_rate);         // Report intrinsic growth rate
    ADREPORT(slow_consumption);    // Report consumption rate for slow-growing corals
    ADREPORT(fast_consumption);    // Report consumption rate for fast-growing corals
    ADREPORT(temp_effect);         // Report temperature effect coefficient
    ADREPORT(immigration_effect);  // Report immigration effect parameter
    ADREPORT(cots_pred);           // Report predicted COTS abundance
    ADREPORT(sd);                  // Report standard deviation of the error term
    
    return nll; // Return negative log likelihood
}

// The following TMB model simulates the dynamics of a plankton ecosystem.
// Equations:
// 1. Nutrient dynamics: N_pred[t+1] = N_pred[t] - nutrient_uptake * P_pred[t] * N_pred[t] + Type(1e-8)
//    - N_pred: Predicted nutrient concentration (g C m^-3)
// 2. Phytoplankton dynamics: P_pred[t+1] = P_pred[t] + growth_rate * P_pred[t] * N_pred[t] - grazing_rate * P_pred[t] * Z_pred[t] + Type(1e-8)
//    - P_pred: Predicted phytoplankton concentration (g C m^-3)
// 3. Zooplankton dynamics: Z_pred[t+1] = Z_pred[t] + assimilation_efficiency * grazing_rate * P_pred[t] * Z_pred[t] - mortality_rate * Z_pred[t] + Type(1e-8)
//    - Z_pred: Predicted zooplankton concentration (g C m^-3)

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  Type nll = 0.0;
  
  // Data inputs: Observed concentrations (g C m^-3) and the number of time steps.
  DATA_VECTOR(N_dat); // observed nutrient concentration
  DATA_VECTOR(P_dat); // observed phytoplankton concentration
  DATA_VECTOR(Z_dat); // observed zooplankton concentration
  DATA_INTEGER(n_time); // number of time steps
  
  // Parameters with smooth penalties to prevent unrealistic values:
  PARAMETER(growth_rate);            // Intrinsic growth rate of phytoplankton (day^-1)
  PARAMETER(mortality_rate);         // Mortality rate of zooplankton (day^-1)
  PARAMETER(grazing_rate);           // Grazing rate of zooplankton (day^-1)
  PARAMETER(nutrient_uptake);        // Nutrient uptake rate by phytoplankton (unitless)
  PARAMETER(assimilation_efficiency); // Efficiency of nutrient assimilation by zooplankton (fraction, unitless)
  
  // Initial state vectors for predicted concentrations
  PARAMETER_VECTOR(N0); // initial nutrient concentrations (g C m^-3)
  PARAMETER_VECTOR(P0); // initial phytoplankton concentrations (g C m^-3)
  PARAMETER_VECTOR(Z0); // initial zooplankton concentrations (g C m^-3)
  
  // Declare prediction vectors
  vector<Type> N_pred(n_time), P_pred(n_time), Z_pred(n_time);
  
  // Loop over time steps to simulate the dynamics
  for(int t = 0; t < n_time; t++){
    if(t == 0){
      N_pred[t] = N0[t];
      P_pred[t] = P0[t];
      Z_pred[t] = Z0[t];
    } else{
      N_pred[t] = N_pred[t-1] - nutrient_uptake * P_pred[t-1] * N_pred[t-1] + Type(1e-8);
      P_pred[t] = P_pred[t-1] + growth_rate * P_pred[t-1] * N_pred[t-1] - grazing_rate * P_pred[t-1] * Z_pred[t-1] + Type(1e-8);
      Z_pred[t] = Z_pred[t-1] + assimilation_efficiency * grazing_rate * P_pred[t-1] * Z_pred[t-1] - mortality_rate * Z_pred[t-1] + Type(1e-8);
    }
  }
  
  // Likelihood calculation:
  // Each observation is modeled using a lognormal distribution with a fixed minimum standard deviation to prevent numerical issues.
  for(int t = 0; t < n_time; t++){
    Type sigma = 0.01; // fixed minimal standard deviation (ensures numerical stability)
    nll += dlnorm(N_dat[t], log(N_pred[t] + Type(1e-8)), sigma, true); // Nutrient likelihood
    nll += dlnorm(P_dat[t], log(P_pred[t] + Type(1e-8)), sigma, true); // Phytoplankton likelihood
    nll += dlnorm(Z_dat[t], log(Z_pred[t] + Type(1e-8)), sigma, true); // Zooplankton likelihood
  }
  
  // Reporting model predictions:
  // Suffix '_pred' indicates predictions corresponding to observed data fields named '_dat'
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);
  
  return nll;
}
#include <TMB.hpp>

// ------------- Model Description -------------
// Equations:
//   (1) Nutrient (N_pred): 
//         N(t+1) = N(t) + dt*( - uptake + recycling )
//   (2) Phytoplankton (P_pred):
//         P(t+1) = P(t) + dt*( growth*P - grazing*P - mortality_P*P )
//   (3) Zooplankton (Z_pred):
//         Z(t+1) = Z(t) + dt*( assimilation*grazing*P - mortality_Z*Z )
// ------------------------------------------------

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  // Data declarations with expected observations (log-transformed) for numerical stability:
  DATA_VECTOR(N_dat);  // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton concentration (g C m^-3)
  DATA_SCALAR(dt);     // Time step (days)

  // Parameter declarations (units given in comments):
  PARAMETER(nutrient_recycling_rate); // Nutrient recycling rate (day^-1)
  PARAMETER(phytoplankton_growth);      // Intrinsic growth rate of phytoplankton (day^-1)
  PARAMETER(zooplankton_grazing_rate);    // Zooplankton grazing rate (day^-1)
  PARAMETER(half_saturation);           // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(mortality_P);               // Phytoplankton mortality rate (day^-1)
  PARAMETER(mortality_Z);               // Zooplankton mortality rate (day^-1)
  PARAMETER(assimilation_efficiency);   // Efficiency of converting grazed phytoplankton into zooplankton biomass (dimensionless)

  // Observation error standard deviations (minimum enforced to prevent numerical issues):
  PARAMETER(sigma_N);
  PARAMETER(sigma_P);
  PARAMETER(sigma_Z);
  
  // Initial conditions for state variables:
  PARAMETER(N0);  // Initial nutrient concentration (g C m^-3)
  PARAMETER(P0);  // Initial phytoplankton concentration (g C m^-3)
  PARAMETER(Z0);  // Initial zooplankton concentration (g C m^-3)

  // Small constant for numerical stability:
  Type eps = Type(1e-8);

  Type nll = 0.0; // negative log likelihood

  // Smooth penalties to constrain parameters within biologically plausible ranges:
  // Using a quadratic penalty when deviating from a soft bound.
  // (i) All rates > 0; (ii) efficiency is between 0 and 1.
  nll += pow( std::min(Type(0), nutrient_recycling_rate - eps), 2 );
  nll += pow( std::min(Type(0), phytoplankton_growth - eps), 2 );
  nll += pow( std::min(Type(0), zooplankton_grazing_rate - eps), 2 );
  nll += pow( std::min(Type(0), half_saturation - eps),2 );
  nll += pow( std::min(Type(0), mortality_P - eps), 2 );
  nll += pow( std::min(Type(0), mortality_Z - eps), 2 );
  nll += pow( std::max(Type(0), assimilation_efficiency - Type(1) ), 2 );
  nll += pow( std::min(Type(0), assimilation_efficiency - eps), 2 );

  // Containers for predictions:
  int n_steps = N_dat.size();
  vector<Type> N_pred(n_steps);
  vector<Type> P_pred(n_steps);
  vector<Type> Z_pred(n_steps);

  // Set initial state:
  N_pred(0) = N0;
  P_pred(0) = P0;
  Z_pred(0) = Z0;

  // Model simulation using a simple Euler integration:
  for(int t = 0; t < n_steps - 1; t++){
    // Equation (1) for Nutrient:
    // uptake = (phytoplankton_growth * N_pred(t)) / (N_pred(t) + half_saturation)
    // recycling = nutrient_recycling_rate * (mortality_P * P_pred(t) + mortality_Z * Z_pred(t))
    Type uptake = (phytoplankton_growth * N_pred(t)) / (N_pred(t) + half_saturation + eps);
    Type recycling = nutrient_recycling_rate * (mortality_P * P_pred(t) + mortality_Z * Z_pred(t));
    N_pred(t+1) = N_pred(t) + dt * (-uptake + recycling);
    
    // Equation (2) for Phytoplankton:
    // growth = phytoplankton_growth * uptake (nutrient-dependent)
    // grazing = zooplankton_grazing_rate * P_pred(t)
    // mortality = mortality_P * P_pred(t)
    Type grazing = zooplankton_grazing_rate * P_pred(t);
    P_pred(t+1) = P_pred(t) + dt * (phytoplankton_growth * uptake - grazing - mortality_P * P_pred(t));
    
    // Equation (3) for Zooplankton:
    // production = assimilation_efficiency * grazing
    // mortality = mortality_Z * Z_pred(t)
    Z_pred(t+1) = Z_pred(t) + dt * (assimilation_efficiency * grazing - mortality_Z * Z_pred(t));
  }

  // Likelihood: compare observations with model predictions using lognormal errors.
  // Applying a fixed lower bound for sigma's.
  Type sigma_N_fixed = sigma_N < eps ? eps : sigma_N;
  Type sigma_P_fixed = sigma_P < eps ? eps : sigma_P;
  Type sigma_Z_fixed = sigma_Z < eps ? eps : sigma_Z;
  
  // Assume that observations have been log-transformed to account for orders of magnitude differences.
  for(int t = 0; t < n_steps; t++){
    // Likelihood contributions (using lognormal density):
    nll -= dlnorm(N_dat(t), log(N_pred(t)+eps), sigma_N_fixed, true);
    nll -= dlnorm(P_dat(t), log(P_pred(t)+eps), sigma_P_fixed, true);
    nll -= dlnorm(Z_dat(t), log(Z_pred(t)+eps), log(Z_pred(t)+eps) , true); // using log(pred) as a placeholder mean
    // Note: Adjust Z likelihood if error distribution needs different structure.
  }

  // Reporting state predictions:
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);

  return nll;
}
#include <TMB.hpp>

// 1. Data inputs and model observations:
//    - nutrient_dat: observed nutrient concentrations (g C m^-3)
//    - phyto_dat: observed phytoplankton concentrations (g C m^-3)
//    - zoo_dat: observed zooplankton concentrations (g C m^-3)
template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(nutrient_dat);  // (g C m^-3)
  DATA_VECTOR(phyto_dat);     // (g C m^-3)
  DATA_VECTOR(zoo_dat);       // (g C m^-3)
  int n = nutrient_dat.size(); // number of time steps

  // 2. Model parameters:
  //    growth_rate: intrinsic growth rate for phytoplankton (day^-1)
  //    assimilation_eff: efficiency of zooplankton assimilating ingested biomass (unitless)
  //    predation_rate: rate at which zooplankton predate phytoplankton (day^-1)
  //    nutrient_uptake: rate of nutrient uptake by phytoplankton (day^-1)
  //    logsd: log of the observational error standard deviation (log-scale)
  PARAMETER(growth_rate);        // (day^-1), initial value from literature or estimate
  PARAMETER(assimilation_eff);     // (unitless), initial estimate/expert opinion
  PARAMETER(predation_rate);       // (day^-1), from literature
  PARAMETER(nutrient_uptake);      // (day^-1), initial estimate
  PARAMETER(logsd);              // error standard deviation on log scale
  Type sd = exp(logsd);          // converting logsd to sd; ensure sd > 0

  // 3. Initialization of state variables (predictions):
  // Variables with suffix _pred are the model predictions for nutrient, phytoplankton, zooplankton.
  vector<Type> nutrient_pred(n);
  vector<Type> phyto_pred(n);
  vector<Type> zoo_pred(n);
  
  // Initialize with first observation (assumed known)
  nutrient_pred[0] = nutrient_dat[0]; // initial nutrient (g C m^-3)
  phyto_pred[0] = phyto_dat[0];       // initial phytoplankton (g C m^-3)
  zoo_pred[0] = zoo_dat[0];           // initial zooplankton (g C m^-3)

  // 4. Dynamical model equations:
  // Equation descriptions:
  //  (1) Nutrient dynamics: nutrient decreases by uptake and is optionally replenished by a small fixed amount.
  //  (2) Phytoplankton dynamics: growth via nutrient uptake (using Holling-type function for smooth transition) and loss by predation.
  //  (3) Zooplankton dynamics: growth driven by ingestion (with assimilation) and natural mortality.
  for (int i = 1; i < n; i++){
    nutrient_pred[i] = nutrient_pred[i-1] - nutrient_uptake * phyto_pred[i-1] + Type(1e-8); // small constant for stability
    phyto_pred[i] = phyto_pred[i-1] 
                    + growth_rate * phyto_pred[i-1] * (nutrient_pred[i-1] / (nutrient_pred[i-1] + Type(1e-8))) 
                    - predation_rate * phyto_pred[i-1] * zoo_pred[i-1]; // smooth uptake and predation interaction
    zoo_pred[i] = zoo_pred[i-1] 
                  + assimilation_eff * predation_rate * phyto_pred[i-1] * zoo_pred[i-1] 
                  - Type(0.1) * zoo_pred[i-1]; // constant mortality rate (day^-1)
  }

  // 5. Likelihood calculation using lognormal error distributions:
  //    A fixed minimum standard deviation is maintained to avoid numerical issues.
  //    The log-transform of model predictions is taken to match the lognormal distribution of strictly positive data.
  Type nll = 0.0;
  for (int i = 0; i < n; i++){
    nll -= dlnorm(nutrient_dat[i], log(nutrient_pred[i] + Type(1e-8)), sd, true);
    nll -= dlnorm(phyto_dat[i], log(phyto_pred[i] + Type(1e-8)), sd, true);
    nll -= dlnorm(zoo_dat[i], log(zoo_pred[i] + Type(1e-8)), sd, true);
  }

  // 6. Reporting model predictions for further diagnostics and analyses:
  REPORT(nutrient_pred);
  REPORT(phyto_pred);
  REPORT(zoo_pred);

  return nll;
}
#include <TMB.hpp>

// 1. Data inputs: observations for nutrient, phytoplankton, and zooplankton concentrations (g C m^-3)
// 2. Parameters for ecological processes are defined below with inline comments indicating their units and origins.
//
// Equations description:
//   (1) Nutrient dynamics: dN/dt = - (growth_rate * (N/(half_saturation + N))) * P + mortality_P * P
//   (2) Phytoplankton dynamics: dP/dt = growth_rate * (N/(half_saturation + N)) * P - grazing_rate * Z * P - mortality_P * P
//   (3) Zooplankton dynamics: dZ/dt = assimilation_efficiency * grazing_rate * Z * P - mortality_Z * Z

template<class Type>
Type objective_function<Type>::operator()()
{
  // DATA: observation vectors; each observation is in g C m^-3.
  DATA_VECTOR(N_dat);      // Nutrient observations
  DATA_VECTOR(P_dat);      // Phytoplankton observations
  DATA_VECTOR(Z_dat);      // Zooplankton observations
  int n = N_dat.size();

  // PARAMETERS (all rates are per day, concentrations in g C m^-3):
  PARAMETER(growth_rate);               // Intrinsic growth rate of phytoplankton (day^-1)
  PARAMETER(half_saturation);           // Half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(grazing_rate);              // Grazing rate of zooplankton on phytoplankton (day^-1)
  PARAMETER(mortality_P);               // Mortality rate of phytoplankton (day^-1)
  PARAMETER(mortality_Z);               // Mortality rate of zooplankton (day^-1)
  PARAMETER(assimilation_efficiency);   // Zooplankton assimilation efficiency (dimensionless, 0-1)

  // Soft penalties for smooth biological constraints (if parameters drift to unrealistic values)
  Type penalty = 0.0;
  // Example: a smooth penalty term (the constant multiplier can be tuned as needed)
  penalty += CppAD::pow(CppAD::log(half_saturation + Type(1e-8)), 2);

  // Standard deviation parameters for the observational error (log-transformed, with a fixed minimum to ensure numerical stability)
  PARAMETER(log_sd_N);  // Log-standard deviation for nutrient observations
  PARAMETER(log_sd_P);  // Log-standard deviation for phytoplankton observations
  PARAMETER(log_sd_Z);  // Log-standard deviation for zooplankton observations
  Type sd_N = exp(log_sd_N) + Type(1e-8);
  Type sd_P = exp(log_sd_P) + Type(1e-8);
  Type sd_Z = exp(log_sd_Z) + Type(1e-8);

  // Initial conditions for the state variables (logged to ensure positivity)
  PARAMETER(log_N0);  // Log initial nutrient concentration (g C m^-3)
  PARAMETER(log_P0);  // Log initial phytoplankton concentration (g C m^-3)
  PARAMETER(log_Z0);  // Log initial zooplankton concentration (g C m^-3)
  Type N0 = exp(log_N0);
  Type P0 = exp(log_P0);
  Type Z0 = exp(log_Z0);

  // Vectors to hold the model predictions for each time step
  vector<Type> N_pred(n), P_pred(n), Z_pred(n);
  N_pred(0) = N0;
  P_pred(0) = P0;
  Z_pred(0) = Z0;

  // Assume a fixed time step (dt = 1 day)
  Type dt = 1.0;
  for(int t = 1; t < n; t++){
    // Calculate uptake and grazing using smooth transitions with small constants for stability:
    Type uptake = growth_rate * (N_pred(t-1) / (half_saturation + N_pred(t-1) + Type(1e-8))) * P_pred(t-1);
    Type growth = growth_rate * (N_pred(t-1) / (half_saturation + N_pred(t-1) + Type(1e-8))) * P_pred(t-1);
    Type grazing = grazing_rate * Z_pred(t-1) * P_pred(t-1);
    Type phyto_mort = mortality_P * P_pred(t-1);
    Type zoo_mort = mortality_Z * Z_pred(t-1);

    // Equation (1): Nutrient dynamics
    N_pred(t) = N_pred(t-1) - uptake * dt + phyto_mort * dt;

    // Equation (2): Phytoplankton dynamics
    P_pred(t) = P_pred(t-1) + (growth - grazing - phyto_mort) * dt;

    // Equation (3): Zooplankton dynamics
    Z_pred(t) = Z_pred(t-1) + (assimilation_efficiency * grazing - zoo_mort) * dt;

    // Ensure predictions remain strictly positive (using a small constant while avoiding hard cutoffs)
    N_pred(t) = fmax(N_pred(t), Type(1e-8));
    P_pred(t) = fmax(P_pred(t), Type(1e-8));
    Z_pred(t) = fmax(Z_pred(t), Type(1e-8));
  }

  // Likelihood calculation using the lognormal distribution:
  // We compare the log-transformed observations and predictions so that all data are strictly positive.
  Type nll = penalty;
  for(int t = 0; t < n; t++){
    nll -= dnorm(log(N_dat(t)), log(N_pred(t)), sd_N, true);
    nll -= dnorm(log(P_dat(t)), log(P_pred(t)), sd_P, true);
    nll -= dnorm(log(Z_dat(t)), log(Z_pred(t)), sd_Z, true);
  }

  // Report predictions and key parameters for monitoring and diagnostics.
  REPORT(N_pred); // Predicted nutrient concentrations
  REPORT(P_pred); // Predicted phytoplankton concentrations
  REPORT(Z_pred); // Predicted zooplankton concentrations
  ADREPORT(growth_rate);
  ADREPORT(half_saturation);
  ADREPORT(grazing_rate);
  ADREPORT(mortality_P);
  ADREPORT(mortality_Z);
  ADREPORT(assimilation_efficiency);

  return nll;
}
#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs: observed variables (all in g C m^-3)
  DATA_VECTOR(nutrient_dat);  // observed nutrient concentrations
  DATA_VECTOR(phyto_dat);     // observed phytoplankton concentrations
  DATA_VECTOR(zoo_dat);       // observed zooplankton concentrations
  int n = nutrient_dat.size(); // number of time steps

  // Model parameters:
  // growth_rate: intrinsic growth rate for phytoplankton (day^-1)
  // assimilation_eff: efficiency of zooplankton assimilating ingested phytoplankton (unitless)
  // predation_rate: rate at which zooplankton predates phytoplankton (day^-1)
  // nutrient_uptake: rate of nutrient uptake by phytoplankton (day^-1)
  // logsd: log-scale observational error standard deviation
  PARAMETER(growth_rate);
  PARAMETER(assimilation_eff);
  PARAMETER(predation_rate);
  PARAMETER(nutrient_uptake);
  PARAMETER(logsd);
  Type sd = exp(logsd);  // error standard deviation

  // Initialize model predictions with the suffix _pred
  vector<Type> nutrient_pred(n);
  vector<Type> phyto_pred(n);
  vector<Type> zoo_pred(n);

  // Initialization: use the first observation as the starting state
  nutrient_pred[0] = nutrient_dat[0];
  phyto_pred[0] = phyto_dat[0];
  zoo_pred[0] = zoo_dat[0];

  // Dynamical model equations:
  // 1. Nutrient dynamics: nutrient decreases by phytoplankton uptake (plus a small constant for stability)
  // 2. Phytoplankton dynamics: growth fueled by nutrient uptake and loss by zooplankton predation
  // 3. Zooplankton dynamics: growth via assimilation of grazed phytoplankton and constant mortality
  for (int i = 1; i < n; i++){
    nutrient_pred[i] = nutrient_pred[i-1] - nutrient_uptake * phyto_pred[i-1] + Type(1e-8);
    phyto_pred[i] = phyto_pred[i-1] + 
                    growth_rate * phyto_pred[i-1] * (nutrient_pred[i-1] / (nutrient_pred[i-1] + Type(1e-8))) -
                    predation_rate * phyto_pred[i-1] * zoo_pred[i-1];
    zoo_pred[i] = zoo_pred[i-1] + 
                  assimilation_eff * predation_rate * phyto_pred[i-1] * zoo_pred[i-1] -
                  Type(0.1) * zoo_pred[i-1]; // constant mortality rate (day^-1)
  }

  // Likelihood calculation using a lognormal error distribution for strictly positive data
  Type nll = 0.0;
  for (int i = 0; i < n; i++){
    nll -= dlnorm(nutrient_dat[i], log(nutrient_pred[i] + Type(1e-8)), sd, true);
    nll -= dlnorm(phyto_dat[i],    log(phyto_pred[i]    + Type(1e-8)), sd, true);
    nll -= dlnorm(zoo_dat[i],      log(zoo_pred[i]      + Type(1e-8)), sd, true);
  }

  // Report predictions for diagnostics
  REPORT(nutrient_pred);
  REPORT(phyto_pred);
  REPORT(zoo_pred);

  return nll;
}
