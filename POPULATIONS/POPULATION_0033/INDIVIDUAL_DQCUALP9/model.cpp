#include <TMB.hpp>

// Crown of Thorns Starfish (COTS) Outbreak Model
// Predicts episodic boom-bust cycles of starfish and their impacts on coral communities

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ----------------------
  // DATA SECTION
  // ----------------------
  DATA_VECTOR(Year);               // Time variable
  DATA_VECTOR(cots_dat);           // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);           // Fast-growing coral cover (% Acropora spp.)
  DATA_VECTOR(slow_dat);           // Slow-growing coral cover (% Faviidae + Porites spp.)
  DATA_VECTOR(sst_dat);            // Sea-Surface Temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);        // COTS larval immigration rate (indv/m2/year)

  // ----------------------
  // PARAMETER SECTION
  // ----------------------
  PARAMETER(log_r_cots);           // Intrinsic growth rate of COTS (log scale), year^-1
  PARAMETER(log_K_cots);           // Carrying capacity of COTS (log scale), ind/m2
  PARAMETER(log_alpha_cots);       // Attack rate of COTS on corals (log scale)
  PARAMETER(log_h_cots);           // Handling time parameter for saturation (log scale)
  PARAMETER(pref_fast);            // Preference weight for fast coral (0–1)
  PARAMETER(log_r_fast);           // Growth rate of fast coral (log scale), % cover/year
  PARAMETER(log_r_slow);           // Growth rate of slow coral (log scale), % cover/year
  PARAMETER(log_K_coral);          // Carrying capacity of total coral cover (log scale), %
  PARAMETER(beta_sst);             // Effect of temperature anomalies on coral growth
  PARAMETER(log_sigma_obs);        // Observation error (log scale), applied equally

  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_cots = exp(log_alpha_cots);
  Type h_cots = exp(log_h_cots);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type K_coral = exp(log_K_coral);
  Type sigma_obs = exp(log_sigma_obs) + Type(1e-8);

  // ----------------------
  // STATE VECTORS
  // ----------------------
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Initialize states using first observed values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Precompute SST mean (since mean() is not available in TMB)
  Type sst_mean = (sum(sst_dat) / Type(sst_dat.size()));

  // ----------------------
  // PROCESS MODEL
  // ----------------------
  for(int t=1; t<n; t++){

    // Total coral cover available
    Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1);

    // --------------------------
    // (1) COTS functional response: Holling II with prey preference
    Type coral_food_prev = pref_fast * fast_pred(t-1) + (Type(1.0) - pref_fast) * slow_pred(t-1);
    Type consumption_rate = (alpha_cots * coral_food_prev) / (Type(1.0) + alpha_cots * h_cots * coral_food_prev);

    // (2) COTS population dynamics with density dependence and immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots);
    Type cots_gain = cots_growth + cotsimm_dat(t);
    Type cots_loss = consumption_rate * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_gain - cots_loss;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8));

    // (3) Coral dynamics (fast and slow groups separately)
    // Environmental modifier from SST: centered at mean
    Type sst_dev = sst_dat(t) - sst_mean;
    Type sst_effect = exp(beta_sst * sst_dev);

    // Fast coral
    Type grow_fast = r_fast * fast_pred(t-1) * (Type(1.0) - coral_total_prev / K_coral) * sst_effect;
    Type loss_fast = alpha_cots * pref_fast * cots_pred(t-1) * (fast_pred(t-1) / (coral_food_prev + Type(1e-8)));
    fast_pred(t) = fast_pred(t-1) + grow_fast - loss_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8));

    // Slow coral
    Type grow_slow = r_slow * slow_pred(t-1) * (Type(1.0) - coral_total_prev / K_coral) * sst_effect;
    Type loss_slow = alpha_cots * (Type(1.0)-pref_fast) * cots_pred(t-1) * (slow_pred(t-1) / (coral_food_prev + Type(1e-8)));
    slow_pred(t) = slow_pred(t-1) + grow_slow - loss_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8));
  }

  // ----------------------
  // LIKELIHOOD
  // ----------------------
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_obs, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_obs, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_obs, true);
  }

  // ----------------------
  // REPORTING
  // ----------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equation summary:
1. Coral consumption by COTS = Holling type II functional response with preference
2. COTS dynamics = logistic growth + larval immigration – resource-dependent mortality
3. Coral dynamics = logistic-like growth constrained by total coral cover and SST modifier – predation loss
*/
