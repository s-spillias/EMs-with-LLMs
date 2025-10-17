#include <TMB.hpp>

/*
Numbered model equations (discrete annual time step, t = 1..T-1; all predictors use t-1 to avoid data leakage)

Let:
- P_t = cots_pred(t) [individuals m^-2]
- F_t = fast_pred(t) [% cover]
- S_t = slow_pred(t) [% cover]
- T_t = sst_dat(t) [°C]
- I_t = cotsimm_dat(t) [individuals m^-2 yr^-1]
- Kspace = carrying capacity for combined coral cover (%)
- eps = 1e-8 small constant

A. Temperature (thermal performance)
1) g_coral(t-1) = exp(-0.5 * ((T_{t-1} - Topt_coral)/Tsd_coral)^2)  ∈ (0,1]
2) g_cots(t-1)  = exp(-0.5 * ((T_{t-1} - Topt_cots )/Tsd_cots )^2)  ∈ (0,1]

B. Space-limited coral growth (saturating with remaining free space; Beverton–Holt-like)
3) Sfree(t-1) = posfun(Kspace - (F_{t-1} + S_{t-1}), eps, pen)     (smoothly ≥ 0; adds penalty if negative)
4) G_F = r_fast * g_coral * F_{t-1} * Sfree / (Sfree + H_fast)
5) G_S = r_slow * g_coral * S_{t-1} * Sfree / (Sfree + H_slow)

C. COTS multi-prey Holling type II with soft preference switching
6) p_fast = invlogit(pref_fast_base_logit + k_switch * ((F_{t-1}/(F_{t-1}+S_{t-1}+eps)) - 0.5))  ∈ (0,1)
   p_slow = 1 - p_fast
7) Den = 1 + a_fast*h_fast*p_fast*F_{t-1} + a_slow*h_slow*p_slow*S_{t-1}
8) Intake per COTS:
   I_F = a_fast * p_fast * F_{t-1} / (Den + eps)
   I_S = a_slow * p_slow * S_{t-1} / (Den + eps)
9) Coral loss to predation:
   L_F = e_fast * P_{t-1} * I_F
   L_S = e_slow * P_{t-1} * I_S

D. Coral non-predation mortality with thermal stress multiplier
10) stress_coral = 1 + stress_coral_mort_coeff * (1 - g_coral)
11) M_F = m_coral_fast * stress_coral * F_{t-1}
    M_S = m_coral_slow * stress_coral * S_{t-1}

E. Coral state updates (soft non-negativity)
12) F_t = posfun(F_{t-1} + G_F - L_F - M_F, eps, pen)
13) S_t = posfun(S_{t-1} + G_S - L_S - M_S, eps, pen)

F. COTS survival, recruitment (food- & temperature-modified), and immigration
14) Survival fraction: surv = exp(-m0 * (1 + stress_cots_mort_coeff*(1 - g_cots))) ∈ (0,1]
15) Density-regulated survivors: P_surv = P_{t-1} * surv / (1 + beta_cots_mort * P_{t-1})
16) Coral food index: food = (F_{t-1} + w_food_slow * S_{t-1}) / (Kspace + eps) ∈ [0,~1]
17) Smooth food threshold for fecundity: f_food = 1 / (1 + exp(-k_food * (food*Kspace - C_food_half)))
18) Larval survival proxy from temperature: f_larv = g_cots
19) Nonlinear Beverton–Holt recruitment:
    P_phi = (P_{t-1} + eps)^{phi_rec}
    R = (alpha_rec * P_phi * f_food * f_larv) / (1 + beta_R * P_phi)
20) Immigration (exogenous forcing): Ieff = k_imm * max(0, I_{t-1})
21) COTS state update: P_t = posfun(P_surv + R + Ieff, eps, pen)

Observation model (for all years, including t=0):
- COTS: lognormal on log(cots): log(cots_dat + eps) ~ Normal(log(cots_pred + eps), sigma_cots)
- Coral cover: logit-normal on proportion (x/100): logit((x + eps)/(100 + 2*eps)) ~ Normal(logit(pred), sigma_fast/slow)

All predicted series initialize at observed values at t=0:
cots_pred(0) = cots_dat(0); fast_pred(0) = fast_dat(0); slow_pred(0) = slow_dat(0)
*/

// Local smooth positivity function equivalent to TMB::posfun to ensure availability across AD types
template<class Type>
Type safe_posfun(const Type& x, const Type& eps, Type& pen) {
  if (x >= eps) return x;                                          // no penalty region
  Type a = eps - x;                                                // amount below eps
  pen += Type(0.01) * a * a;                                       // smooth quadratic penalty
  return eps / (Type(2.0) - x / eps);                              // smooth lower bound transform
}

template<class Type>
Type objective_function<Type>::operator()() {
  Type nll = 0.0;                      // Negative log-likelihood accumulator
  Type pen = 0.0;                      // Smooth penalties accumulator
  const Type eps = Type(1e-8);         // Small constant for numerical stability

  // DATA: Time and observed response/forcing series (units in comments)
  DATA_VECTOR(Year);                   // Year [year]; used for alignment/reporting
  DATA_VECTOR(cots_dat);               // Adult COTS abundance [individuals m^-2]
  DATA_VECTOR(fast_dat);               // Fast-growing coral cover [% of area]
  DATA_VECTOR(slow_dat);               // Slow-growing coral cover [% of area]
  DATA_VECTOR(sst_dat);                // Sea-surface temperature [°C]
  DATA_VECTOR(cotsimm_dat);            // COTS larval immigration rate [individuals m^-2 yr^-1]

  const int n = Year.size();           // Time series length [years]

  // PARAMETERS (unconstrained forms; transformed below). Each line includes description and units.

  // Coral growth and space limitation
  PARAMETER(log_r_fast);               // ln(year^-1): Intrinsic growth rate of fast coral (Acropora). Initial from literature/estimates.
  PARAMETER(log_r_slow);               // ln(year^-1): Intrinsic growth rate of slow coral (Faviidae/Porites).
  PARAMETER(log_H_fast);               // ln(%): Half-saturation of space limitation for fast coral.
  PARAMETER(log_H_slow);               // ln(%): Half-saturation of space limitation for slow coral.
  PARAMETER(K_space_logit);            // logit on [60,100]%: Total coral carrying capacity (combined fast+slow).

  // Coral background mortality
  PARAMETER(log_m_coral_fast);         // ln(year^-1): Background mortality rate of fast coral (excludes COTS predation).
  PARAMETER(log_m_coral_slow);         // ln(year^-1): Background mortality rate of slow coral (excludes COTS predation).
  PARAMETER(log_stress_coral_mort_coeff); // ln(dimensionless): Multiplier of coral mortality per unit thermal stress.

  // COTS functional response and prey-specific efficiencies
  PARAMETER(log_a_fast);               // ln((% cover)^-1 year^-1): Attack rate on fast coral.
  PARAMETER(log_a_slow);               // ln((% cover)^-1 year^-1): Attack rate on slow coral.
  PARAMETER(log_h_fast);               // ln(year): Handling time weight for fast coral (scaled to % units).
  PARAMETER(log_h_slow);               // ln(year): Handling time weight for slow coral (scaled to % units).
  PARAMETER(log_e_fast);               // ln(% cover per (ind m^-2) per unit intake): Efficiency mapping intake to fast coral loss.
  PARAMETER(log_e_slow);               // ln(% cover per (ind m^-2) per unit intake): Efficiency mapping intake to slow coral loss.
  PARAMETER(pref_fast_base_logit);     // logit(pref): Baseline preference for fast coral (before switching).
  PARAMETER(log_k_switch);             // ln(dimensionless): Strength of preference switching as fast vs slow balance changes.

  // COTS survival, density dependence, and recruitment
  PARAMETER(log_m0);                   // ln(year^-1): Baseline instantaneous adult COTS mortality rate.
  PARAMETER(log_stress_cots_mort_coeff); // ln(dimensionless): Mortality stress multiplier per unit thermal stress for COTS.
  PARAMETER(log_beta_cots_mort);       // ln((ind m^-2)^-1): Density-dependent mortality strength for adults.
  PARAMETER(log_alpha_rec);            // ln((ind m^-2)^(1-phi) year^-1): Max recruitment scale parameter.
  PARAMETER(phi_rec_unbounded);        // unconstrained: Nonlinearity exponent phi in (0.5, 3), mapped smoothly via logistic.
  PARAMETER(log_beta_R);               // ln((ind m^-2)^-phi): Beverton–Holt recruitment density regulation parameter.
  PARAMETER(log_k_food);               // ln((% cover)^-1): Steepness of food threshold for fecundity.
  PARAMETER(C_food_half_logit);        // logit on [0,100]%: Half-saturation (in % cover) for fecundity vs. food index.
  PARAMETER(w_food_slow_logit);        // logit on [0,1]: Weight of slow coral in food index (fast weight = 1).

  // Environmental temperature response parameters
  PARAMETER(Topt_coral);               // °C: Coral growth temperature optimum.
  PARAMETER(log_Tsd_coral);            // ln(°C): Coral growth thermal breadth (std dev of Gaussian).
  PARAMETER(Topt_cots);                // °C: COTS survival/larval survival temperature optimum.
  PARAMETER(log_Tsd_cots);             // ln(°C): COTS thermal breadth (std dev of Gaussian).

  // Immigration and observation error
  PARAMETER(log_k_imm);                // ln(year): Scaling of larval immigration forcing to adult recruits (per year step).
  PARAMETER(log_sigma_obs_cots);       // ln: Observation SD for log(COTS).
  PARAMETER(log_sigma_obs_fast);       // ln: Observation SD for logit(fast cover proportion).
  PARAMETER(log_sigma_obs_slow);       // ln: Observation SD for logit(slow cover proportion).

  // Transform parameters to natural scale (with smooth bounds where needed)
  Type r_fast  = exp(log_r_fast);                       // year^-1
  Type r_slow  = exp(log_r_slow);                       // year^-1
  Type H_fast  = exp(log_H_fast) + eps;                 // %
  Type H_slow  = exp(log_H_slow) + eps;                 // %
  // Map K_space to [60,100] using logistic
  Type K_space = Type(60.0) + invlogit(K_space_logit) * Type(40.0);  // %

  Type m_coral_fast = exp(log_m_coral_fast);            // year^-1
  Type m_coral_slow = exp(log_m_coral_slow);            // year^-1
  Type stress_coral_mort_coeff = exp(log_stress_coral_mort_coeff); // dimensionless

  Type a_fast = exp(log_a_fast);                        // (%^-1 year^-1)
  Type a_slow = exp(log_a_slow);                        // (%^-1 year^-1)
  Type h_fast = exp(log_h_fast);                        // year
  Type h_slow = exp(log_h_slow);                        // year
  Type e_fast = exp(log_e_fast);                        // % per (ind m^-2) per intake
  Type e_slow = exp(log_e_slow);                        // % per (ind m^-2) per intake
  Type k_switch = exp(log_k_switch);                    // dimensionless

  Type m0 = exp(log_m0);                                // year^-1
  Type stress_cots_mort_coeff = exp(log_stress_cots_mort_coeff); // dimensionless
  Type beta_cots_mort = exp(log_beta_cots_mort);        // (ind m^-2)^-1
  Type alpha_rec = exp(log_alpha_rec);                  // (ind m^-2)^(1-phi) year^-1
  // Smooth bounds for phi in [0.5, 3.0]
  const Type phi_lo = Type(0.5);
  const Type phi_hi = Type(3.0);
  Type phi_rec = phi_lo + invlogit(phi_rec_unbounded) * (phi_hi - phi_lo); // dimensionless
  Type beta_R = exp(log_beta_R);                        // (ind m^-2)^-phi
  Type k_food = exp(log_k_food);                        // (% cover)^-1
  // Map C_food_half to [0,100]
  Type C_food_half = invlogit(C_food_half_logit) * Type(100.0); // %
  Type w_food_slow = invlogit(w_food_slow_logit);       // dimensionless in (0,1)

  Type Tsd_coral = exp(log_Tsd_coral);                  // °C
  Type Tsd_cots  = exp(log_Tsd_cots);                   // °C

  Type k_imm = exp(log_k_imm);                          // year

  // Observation SDs with minimum floors for numerical stability
  Type sigma_cots = exp(log_sigma_obs_cots);            // SD on log scale
  Type sigma_fast = exp(log_sigma_obs_fast);            // SD on logit scale
  Type sigma_slow = exp(log_sigma_obs_slow);            // SD on logit scale
  const Type sigma_min_log = Type(0.05);                // minimum SD floor (log / logit)
  sigma_cots = sqrt(sigma_cots * sigma_cots + sigma_min_log * sigma_min_log);
  sigma_fast = sqrt(sigma_fast * sigma_fast + sigma_min_log * sigma_min_log);
  sigma_slow = sqrt(sigma_slow * sigma_slow + sigma_min_log * sigma_min_log);

  // Predictions (initialize at observed initial conditions to avoid data leakage)
  vector<Type> cots_pred(n);                            // predicted COTS [ind m^-2]
  vector<Type> fast_pred(n);                            // predicted fast coral [%]
  vector<Type> slow_pred(n);                            // predicted slow coral [%]
  cots_pred(0) = cots_dat(0);                           // initialization from data
  fast_pred(0) = fast_dat(0);                           // initialization from data
  slow_pred(0) = slow_dat(0);                           // initialization from data

  // Time loop (forward simulation; use t-1 states)
  for (int t = 1; t < n; ++t) {
    // Previous states
    Type P_prev = cots_pred(t - 1) + eps;               // adults [ind m^-2]
    Type F_prev = fast_pred(t - 1) + eps;               // fast coral [%]
    Type S_prev = slow_pred(t - 1) + eps;               // slow coral [%]

    // Forcing at t-1
    Type T_prev = sst_dat(t - 1);                       // SST [°C]
    Type I_prev = cotsimm_dat(t - 1);                   // immigration [ind m^-2 yr^-1]

    // Thermal performance modifiers
    Type g_coral = exp(-Type(0.5) * pow((T_prev - Topt_coral) / (Tsd_coral + eps), 2)); // (0,1]
    Type g_cots  = exp(-Type(0.5) * pow((T_prev - Topt_cots ) / (Tsd_cots  + eps), 2)); // (0,1]

    // Free space (smoothly non-negative)
    Type Sfree_raw = K_space - (F_prev + S_prev);
    Type Sfree = safe_posfun(Sfree_raw, eps, pen);      // % available

    // Coral growth (space-limited; Beverton–Holt-like)
    Type G_F = r_fast * g_coral * F_prev * Sfree / (Sfree + H_fast + eps);
    Type G_S = r_slow * g_coral * S_prev * Sfree / (Sfree + H_slow + eps);

    // Preference switching toward slow coral when fast depleted (softly)
    Type coral_total_prev = F_prev + S_prev;
    Type frac_fast = F_prev / (coral_total_prev + eps); // fraction ∈ (0,1)
    Type p_fast = invlogit(pref_fast_base_logit + k_switch * (frac_fast - Type(0.5))); // ∈ (0,1)
    Type p_slow = Type(1.0) - p_fast;

    // Multi-prey Holling type II intake per COTS
    Type Den = Type(1.0) + a_fast * h_fast * p_fast * F_prev + a_slow * h_slow * p_slow * S_prev;
    Type I_F = a_fast * p_fast * F_prev / (Den + eps);
    Type I_S = a_slow * p_slow * S_prev / (Den + eps);

    // Coral loss due to COTS predation (process-specific efficiencies)
    Type L_F = e_fast * P_prev * I_F;
    Type L_S = e_slow * P_prev * I_S;

    // Non-predation coral mortality with thermal stress
    Type stress_coral = Type(1.0) + stress_coral_mort_coeff * (Type(1.0) - g_coral);
    Type M_F = m_coral_fast * stress_coral * F_prev;
    Type M_S = m_coral_slow * stress_coral * S_prev;

    // Coral updates (smooth non-negativity; sums can exceed K_space but are penalized through posfun usage)
    Type F_next = safe_posfun(F_prev + G_F - L_F - M_F, eps, pen);
    Type S_next = safe_posfun(S_prev + G_S - L_S - M_S, eps, pen);

    // COTS survival with thermal stress and density dependence
    Type surv = exp(-m0 * (Type(1.0) + stress_cots_mort_coeff * (Type(1.0) - g_cots))); // (0,1]
    Type P_surv = P_prev * surv / (Type(1.0) + beta_cots_mort * P_prev);

    // Food-dependent fecundity (weight slow coral contribution)
    Type food_index = (F_prev + w_food_slow * S_prev) / (K_space + eps); // 0..~1
    Type f_food = Type(1.0) / (Type(1.0) + exp(-k_food * (food_index * K_space - C_food_half)));

    // Temperature-modified larval survival (proxy)
    Type f_larv = g_cots;

    // Nonlinear Beverton–Holt recruitment
    Type P_phi = pow(P_prev + eps, phi_rec);
    Type R = (alpha_rec * P_phi * f_food * f_larv) / (Type(1.0) + beta_R * P_phi);

    // Immigration forcing (ensure non-negative with soft clip via safe_posfun)
    Type Ieff = k_imm * safe_posfun(I_prev, eps, pen);

    // COTS update (smooth non-negativity)
    Type P_next = safe_posfun(P_surv + R + Ieff, eps, pen);

    // Save predictions
    fast_pred(t) = F_next;
    slow_pred(t) = S_next;
    cots_pred(t) = P_next;
  }

  // Observation likelihoods for all time steps (including t=0)
  for (int t = 0; t < n; ++t) {
    // COTS: lognormal
    Type c_obs = log(cots_dat(t) + eps);
    Type c_hat = log(cots_pred(t) + eps);
    nll -= dnorm(c_obs, c_hat, sigma_cots, true);

    // Coral cover: logit-normal on proportions, keep strictly within (0,1) via eps trick
    Type f_obs_p = (fast_dat(t) + eps) / (Type(100.0) + Type(2.0) * eps);
    Type s_obs_p = (slow_dat(t) + eps) / (Type(100.0) + Type(2.0) * eps);
    Type f_hat_p = (fast_pred(t) + eps) / (Type(100.0) + Type(2.0) * eps);
    Type s_hat_p = (slow_pred(t) + eps) / (Type(100.0) + Type(2.0) * eps);

    Type f_obs_logit = log(f_obs_p / (Type(1.0) - f_obs_p));
    Type s_obs_logit = log(s_obs_p / (Type(1.0) - s_obs_p));
    Type f_hat_logit = log(f_hat_p / (Type(1.0) - f_hat_p));
    Type s_hat_logit = log(s_hat_p / (Type(1.0) - s_hat_p));

    nll -= dnorm(f_obs_logit, f_hat_logit, sigma_fast, true);
    nll -= dnorm(s_obs_logit, s_hat_logit, sigma_slow, true);
  }

  // Add any accumulated smooth penalties
  nll += pen;

  // Report predictions (for diagnostics and downstream use)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
