#include <TMB.hpp>

// Crown of Thorns Outbreak Model - Great Barrier Reef
// Captures COTS boom-bust cycles and selective predation on corals.

// DATA AND PARAMETERS
template<class Type>
Type objective_function<Type>::operator()() {
  
  // DATA ------------------------
  DATA_VECTOR(Year);        // numeric year
  DATA_VECTOR(cots_dat);    // observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);    // observed Acropora coral cover (%)
  DATA_VECTOR(slow_dat);    // observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // observed sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // larval immigration events (individuals/m2/year)

  int nT = Year.size();

  // PARAMETERS ------------------
  PARAMETER(r_cots);        // intrinsic growth rate of COTS (year^-1)
  PARAMETER(K_cots);        // carrying capacity of COTS (individuals/m2)
  PARAMETER(mort_cots);     // background mortality rate of COTS (year^-1)
  PARAMETER(alpha_acropora);// predation rate of COTS on fast-growing coral (% cover per ind per year)
  PARAMETER(alpha_slow);    // predation rate of COTS on slow-growing coral (% cover per ind per year)
  PARAMETER(r_fast);        // intrinsic growth rate of Acropora coral (% per year)
  PARAMETER(r_slow);        // intrinsic growth rate of slow-growing coral (% per year)
  PARAMETER(temp_effect);   // temperature sensitivity of COTS growth (per °C above baseline)
  PARAMETER(log_sigma_cots);// log SD for COTS obs likelihood
  PARAMETER(log_sigma_coral);// log SD for coral obs likelihood

  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);

  // STATE VARIABLES -------------
  vector<Type> cots_pred(nT);
  vector<Type> fast_pred(nT);
  vector<Type> slow_pred(nT);

  // INITIAL CONDITIONS from first observation
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // DYNAMICS --------------------
  for(int t=1; t<nT; t++){

    // (1) Environmental modification of growth rate
    Type temp_dev = sst_dat(t-1) - Type(27.0); // baseline temperature ~27 C
    Type r_eff = r_cots + temp_effect * temp_dev;

    // (2) COTS population growth (logistic + immigration)
    Type cots_growth = r_eff * cots_pred(t-1) * (1 - cots_pred(t-1) / (K_cots + Type(1e-8)));
    Type cots_next = cots_pred(t-1) + cots_growth 
                      - mort_cots * cots_pred(t-1) 
                      + cotsimm_dat(t-1);

    cots_next = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8));
    cots_pred(t) = cots_next;

    // (3) Coral dynamics
    // Fast coral: logistic growth minus COTS predation
    Type fast_growth = r_fast * fast_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1)) / 100.0);
    Type fast_loss = alpha_acropora * cots_pred(t-1) * fast_pred(t-1);
    fast_pred(t) = fmax(Type(1e-8), fast_pred(t-1) + fast_growth - fast_loss);

    // Slow coral: logistic growth minus COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (1 - (fast_pred(t-1) + slow_pred(t-1)) / 100.0);
    Type slow_loss = alpha_slow * cots_pred(t-1) * slow_pred(t-1);
    slow_pred(t) = fmax(Type(1e-8), slow_pred(t-1) + slow_growth - slow_loss);
  }

  // LIKELIHOOD ------------------
  Type nll = 0.0;

  for(int t=0; t<nT; t++){
    // Lognormal likelihood for COTS (always positive densities)
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);

    // Normal likelihood for coral cover (%)
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_coral, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_coral, true);
  }

  // REPORT ----------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
