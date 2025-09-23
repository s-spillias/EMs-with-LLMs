#include <TMB.hpp>

// Crown-of-Thorns starfish outbreak model
// Ecological mechanisms: boom-bust cycles driven by reproduction, immigration, coral depletion
// Includes selective predation (Acropora >> Faviidae/Porites), SST effect, nonlinear responses

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA ---
  DATA_VECTOR(Year);        // year identifier
  DATA_VECTOR(cots_dat);    // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (C)
  DATA_VECTOR(cotsimm_dat); // Larval immigration (individuals/m2/year)

  // --- PARAMETERS ---
  PARAMETER(r_cots);         // Intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);         // Carrying capacity of COTS (ind/m2)
  PARAMETER(alpha_fast);     // Attack rate on fast-growing corals
  PARAMETER(alpha_slow);     // Attack rate on slow-growing corals
  PARAMETER(handling_time);  // Handling time for Holling II response
  PARAMETER(sst_effect);     // Sensitivity of COTS survival to SST
  PARAMETER(immigration_rate); // Scaling for larval immigration
  PARAMETER(growth_fast);    // Growth rate of Acropora corals (% cover/year)
  PARAMETER(growth_slow);    // Growth rate of slow corals (% cover/year)
  PARAMETER(sigma_obs);      // Observation error

  int n = Year.size();

  // --- STATE VARIABLES (predictions) ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- INITIAL CONDITIONS ---
  cots_pred(0) = cots_dat(0); // start from observed
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // --- DYNAMICS ---
  Type eps = Type(1e-8); // small constant

  for(int t=1; t<n; t++){
    // Previous state
    Type C_prev = cots_pred(t-1);
    Type F_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);

    // Functional response: per capita predation on fast and slow coral (Holling II)
    Type denom = (Type(1.0) + handling_time * (alpha_fast*F_prev + alpha_slow*S_prev) + eps);
    Type cons_fast = (alpha_fast * F_prev) / denom;
    Type cons_slow = (alpha_slow * S_prev) / denom;

    // Effective survival adjustment from SST
    Type sst_dev = sst_dat(t-1) - Type(27.0); // deviation from reference temp (27C)
    Type survival_mod = exp(sst_effect * sst_dev);

    // COTS dynamics: logistic + immigration + SST-dependent survival
    Type C_growth = r_cots * C_prev * (Type(1.0) - C_prev/K_cots);
    Type C_imm = immigration_rate * cotsimm_dat(t-1);
    cots_pred(t) = C_prev + (C_growth + C_imm) * survival_mod;
    if(cots_pred(t) < eps) cots_pred(t) = eps;

    // Coral dynamics: growth minus predation losses
    Type F_growth = growth_fast * F_prev * (Type(1.0) - F_prev/Type(100.0));
    Type S_growth = growth_slow * S_prev * (Type(1.0) - S_prev/Type(100.0));
    Type F_loss = cons_fast * C_prev;
    Type S_loss = cons_slow * C_prev;

    fast_pred(t) = F_prev + F_growth - F_loss;
    slow_pred(t) = S_prev + S_growth - S_loss;

    if(fast_pred(t) < eps) fast_pred(t) = eps;
    if(slow_pred(t) < eps) slow_pred(t) = eps;
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal observational error
    Type sd = sigma_obs + eps;
    nll -= dnorm(log(cots_dat(t)+eps), log(cots_pred(t)+eps), sd, true);
    nll -= dnorm(log(fast_dat(t)+eps), log(fast_pred(t)+eps), sd, true);
    nll -= dnorm(log(slow_dat(t)+eps), log(slow_pred(t)+eps), sd, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equations:
1. COTS[t] = C_prev + (r*C_prev*(1-C_prev/K) + immigration*larval) * survival(SST)
2. Survival(SST) = exp(sst_effect*(SST-27))
3. Consumption rates = (alpha*coral)/(1+handling*(alpha_fast*fast + alpha_slow*slow))
4. Fast coral[t] = F_prev + growth_fast*F_prev*(1-F_prev/100) - cons_fast*C_prev
5. Slow coral[t] = S_prev + growth_slow*S_prev*(1-S_prev/100) - cons_slow*C_prev
6. Likelihood: lognormal errors on all observed states
*/
