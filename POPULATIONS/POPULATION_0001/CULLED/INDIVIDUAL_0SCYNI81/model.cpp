#include <TMB.hpp> // TMB header providing automatic differentiation

// Template Model Builder model for episodic COTS outbreaks with coral predation dynamics.
template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS:
  // Year vector (time steps) - from the data file, e.g., "Year"
  DATA_VECTOR(Year); // Observation years (e.g., 1980, 1981, ...)
  DATA_VECTOR(cots_dat); // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Observed cover of fast-growing coral (Acropora spp.; in %)
  DATA_VECTOR(slow_dat); // Observed cover of slow-growing coral (Faviidae spp. and Porites spp.; in %)

  int n = Year.size();
  
  // PARAMETERS:
  // 1. r: intrinsic growth rate of COTS (year^-1)
  // 2. K: carrying capacity of COTS (individuals per m^2)
  // 3. alpha_fast: predation rate on fast-growing coral (% loss per individual COTS per year)
  // 4. alpha_slow: predation rate on slow-growing coral (% loss per individual COTS per year)
  // 5. beta: efficiency for converting coral loss into COTS growth (unitless)
  // 6. log_sigma: log error standard deviation for observation process
  PARAMETER(r);              
  PARAMETER(K);              
  PARAMETER(alpha_fast);     
  PARAMETER(alpha_slow);     
  PARAMETER(beta);
  PARAMETER(H_fast);
  PARAMETER(H_slow);
  PARAMETER(log_sigma);      
  
  // Small constant for numerical stability
  Type eps = Type(1e-8);
  
  // INITIALIZE PREDICTIONS:
  // We use the first observation as the initial condition (allowed for initialization only)
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  cots_pred[0] = cots_dat[0];
  fast_pred[0] = fast_dat[0];
  slow_pred[0] = slow_dat[0];
  
  // Numbered Equations for Model Dynamics:
  // Equation 1: COTS dynamics - Logistic growth with losses due to coral predation:
  //   cots[t] = cots[t-1] + r * cots[t-1] * (1 - cots[t-1]/K) - (alpha_fast * fast_pred[t-1] + alpha_slow * slow_pred[t-1]) * cots[t-1]
  // Equation 2: Fast-growing coral dynamics - Decline due to predation by COTS:
  //   fast[t] = fast[t-1] - beta * alpha_fast * cots[t-1] * fast_pred[t-1]
  // Equation 3: Slow-growing coral dynamics - Decline due to predation by COTS:
  //   slow[t] = slow[t-1] - beta * alpha_slow * cots[t-1] * slow_pred[t-1]
  for(int t = 1; t < n; t++){
    // Smooth transition using saturating functional response to avoid hard thresholds
    Type cots_growth = r * cots_pred[t-1] * (1.0 - cots_pred[t-1] / (K + eps));
    Type predation_loss = (alpha_fast * fast_pred[t-1] + alpha_slow * slow_pred[t-1]) * cots_pred[t-1];
    
    cots_pred[t] = cots_pred[t-1] + cots_growth - predation_loss;
    
    // Coral dynamics with a lower bound to ensure values do not become negative
    fast_pred[t] = fast_pred[t-1] - beta * alpha_fast * cots_pred[t-1] * (fast_pred[t-1]*fast_pred[t-1]) / (H_fast + fast_pred[t-1]*fast_pred[t-1]);
    if(fast_pred[t] < eps) fast_pred[t] = eps;
    slow_pred[t] = slow_pred[t-1] - beta * alpha_slow * cots_pred[t-1] * (slow_pred[t-1]*slow_pred[t-1]) / (H_slow + slow_pred[t-1]*slow_pred[t-1]);
    if(slow_pred[t] < eps) slow_pred[t] = eps;
  }
  
  // LIKELIHOOD CALCULATION:
  // Using lognormal likelihood for strictly positive data.
  Type sigma = exp(log_sigma);
  Type nll = 0.0;
  
  // Loop through each time step and contribute to the likelihood
  for (int t=0; t<n; t++){
    // Using log-transformation to account for multiplicative error
    nll -= dnorm(log(cots_dat[t]+eps), log(cots_pred[t]+eps), sigma, true);
    nll -= dnorm(log(fast_dat[t]+eps), log(fast_pred[t]+eps), sigma, true);
    nll -= dnorm(log(slow_dat[t]+eps), log(slow_pred[t]+eps), sigma, true);
  }
  
  // REPORT model predictions
  REPORT(cots_pred); // Predicted COTS abundance time series (_pred)
  REPORT(fast_pred); // Predicted fast-growing coral cover time series (_pred)
  REPORT(slow_pred); // Predicted slow-growing coral cover time series (_pred)
  
  return nll;
}
