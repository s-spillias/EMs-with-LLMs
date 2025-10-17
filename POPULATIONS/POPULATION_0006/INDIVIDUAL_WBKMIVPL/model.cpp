#include <TMB.hpp>  // TMB model header

/* 
Equations (yearly time step; all transitions use previous-step states only):

(1) Multi-prey Holling type II per-capita predation mortality on corals:
    D_t = 1 + h_A * a_A * A_t + h_S * a_S * S_t
    m_predA_t = C_t * a_A / D_t
    m_predS_t = C_t * a_S / D_t
    where A_t, S_t are percent cover (%), C_t is COTS density (ind m^-2);
    a_* are attack rates; h_* are handling-time scalars.

(2) Coral temperature modifiers and bleaching:
    g_TA_t = exp( -0.5 * ((T_t - T_Aopt)/T_Asd)^2 )
    g_TS_t = exp( -0.5 * ((T_t - T_Sopt)/T_Ssd)^2 )
    b_t = 1 / (1 + exp( -k_bleach * (T_t - T_bleach) ))
    m_bleachA_t = b_bleach_A * m_bleach_max * b_t
    m_bleachS_t = b_bleach_S * m_bleach_max * b_t

(3) Coral updates (multiplicative for positivity, plus density-independent recruitment):
    A_{t+1} = A_t * exp( g_TA_t * r_A * (1 - (A_t + S_t)/K_reef) - m_predA_t - m_bleachA_t ) 
              + rec_A * (1 - (A_t + S_t)/K_reef)
    S_{t+1} = S_t * exp( g_TS_t * r_S * (1 - (A_t + S_t)/K_reef) - m_predS_t - m_bleachS_t ) 
              + rec_S * (1 - (A_t + S_t)/K_reef)

(4) Food availability (preference-weighted, saturating):
    F_t = (p_A * A_t + p_S * S_t) / (K_food + A_t + S_t + eps)

(5) COTS temperature performance and Allee effect:
    f_T_t = exp( -0.5 * ((T_t - T_opt)/T_sd)^2 )
    g_Allee_t = 1 / (1 + exp( -k_allee * (C_t - C_allee) ))

(6) COTS recruitment, juvenile dynamics, and adult survival (boom–bust via Ricker + crowding + maturation delay):
    R_t = alpha_C * C_t * exp( -beta_C * C_t ) * (eps_R * F_t) * f_T_t * g_Allee_t + I_t
    J_{t+1} = (1 - m_JA) * J_t * exp(-mu_J) + R_t
    C_{t+1} = C_t * exp( -mu_C - q_C * C_t ) + m_JA * J_t

Observation model (lognormal, all observations included):
    y ~ LogNormal( log(pred + eps), sigma )
with sigma floored at sigma_min to ensure stability.

Initial conditions:
  - Coral and adult COTS states initialized from the first observations (A_0 = fast_dat(0), S_0 = slow_dat(0), C_0 = cots_dat(0)).
  - Juveniles initialized from parameter J0 (unobserved state).
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::CondExpLt; // conditional expressions for smooth-ish penalties
  using CppAD::CondExpGt;

  // -----------------------------
  // Data
  // -----------------------------
  DATA_VECTOR(Year);        // calendar year of each observation (year)
  DATA_VECTOR(cots_dat);    // observed adult COTS density (ind m^-2)
  DATA_VECTOR(fast_dat);    // observed fast-growing coral cover (Acropora) (% cover)
  DATA_VECTOR(slow_dat);    // observed slow-growing coral cover (Faviidae/Porites) (% cover)
  DATA_VECTOR(sst_dat);     // observed sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // observed larval immigration rate (ind m^-2 yr^-1)

  int n = Year.size();      // number of time steps (years)

  // -----------------------------
  // Parameters (all free; bounds penalized smoothly)
  // -----------------------------
  PARAMETER(alpha_C);      // alpha_C: COTS reproductive productivity scaling (yr^-1); informed by literature on fecundity and survivorship
  PARAMETER(beta_C);       // beta_C: COTS Ricker density dependence (m^2 ind^-1); controls downturn at high adult density; initial estimate
  PARAMETER(mu_C);         // mu_C: COTS baseline adult mortality (yr^-1); literature/initial estimate
  PARAMETER(q_C);          // q_C: crowding mortality coefficient (m^2 ind^-1 yr^-1); enforces bust post-boom; initial estimate

  // Juvenile-stage parameters
  PARAMETER(mu_J);         // mu_J: juvenile baseline mortality (yr^-1)
  PARAMETER(m_JA);         // m_JA: fraction of juveniles maturing to adults each year (0-1)
  PARAMETER(J0);           // J0: initial juvenile density (ind m^-2)

  PARAMETER(p_A);          // p_A: preference weight for Acropora in reproduction (dimensionless [0,1]); literature/initial estimate
  PARAMETER(p_S);          // p_S: preference weight for slow corals in reproduction (dimensionless [0,1]); literature/initial estimate
  PARAMETER(K_food);       // K_food: half-saturation of food effect (percent cover); initial estimate
  PARAMETER(eps_R);        // eps_R: efficiency mapping food index to recruits (dimensionless 0-1); literature/initial estimate

  PARAMETER(k_allee);      // k_allee: slope of Allee effect sigmoid (m^2 ind^-1); initial estimate
  PARAMETER(C_allee);      // C_allee: Allee midpoint density (ind m^-2); initial estimate

  PARAMETER(T_opt);        // T_opt: COTS recruitment temperature optimum (deg C); literature/initial estimate
  PARAMETER(T_sd);         // T_sd: COTS recruitment temperature breadth (deg C); literature/initial estimate

  PARAMETER(r_A);          // r_A: intrinsic growth rate of Acropora (yr^-1); literature/initial estimate
  PARAMETER(r_S);          // r_S: intrinsic growth rate of slow corals (yr^-1); literature/initial estimate
  PARAMETER(K_reef);       // K_reef: total reef space capacity (% cover); initial estimate

  PARAMETER(rec_A);        // rec_A: density-independent Acropora recruitment (percent cover yr^-1); initial estimate
  PARAMETER(rec_S);        // rec_S: density-independent slow-coral recruitment (percent cover yr^-1); initial estimate

  PARAMETER(a_A);          // a_A: COTS attack rate on Acropora (ind^-1 yr^-1 per % cover); literature/initial estimate
  PARAMETER(a_S);          // a_S: COTS attack rate on slow corals (ind^-1 yr^-1 per % cover); literature/initial estimate
  PARAMETER(h_A);          // h_A: handling-time scalar for Acropora (yr per % cover unit); initial estimate
  PARAMETER(h_S);          // h_S: handling-time scalar for slow corals (yr per % cover unit); initial estimate

  PARAMETER(T_Aopt);       // T_Aopt: Acropora growth temperature optimum (deg C); literature/initial estimate
  PARAMETER(T_Asd);        // T_Asd: Acropora thermal breadth for growth (deg C); literature/initial estimate
  PARAMETER(T_Sopt);       // T_Sopt: slow-coral growth temperature optimum (deg C); literature/initial estimate
  PARAMETER(T_Ssd);        // T_Ssd: slow-coral thermal breadth for growth (deg C); literature/initial estimate
  PARAMETER(T_bleach);     // T_bleach: bleaching inflection temperature (deg C); literature/initial estimate
  PARAMETER(k_bleach);     // k_bleach: steepness of bleaching ramp (deg C^-1); literature/initial estimate
  PARAMETER(m_bleach_max); // m_bleach_max: maximum bleaching mortality (yr^-1); literature/initial estimate
  PARAMETER(b_bleach_A);   // b_bleach_A: Acropora bleaching sensitivity multiplier (dimensionless); literature/initial estimate
  PARAMETER(b_bleach_S);   // b_bleach_S: slow-coral bleaching sensitivity multiplier (dimensionless); literature/initial estimate

  PARAMETER(sigma_cots);   // sigma_cots: lognormal observation SD for COTS (dimensionless log-scale); initial estimate
  PARAMETER(sigma_fast);   // sigma_fast: lognormal observation SD for Acropora cover (dimensionless log-scale); initial estimate
  PARAMETER(sigma_slow);   // sigma_slow: lognormal observation SD for slow coral cover (dimensionless log-scale); initial estimate

  // -----------------------------
  // Numerical constants
  // -----------------------------
  Type eps = Type(1e-8);   // small constant to avoid division by zero
  Type two_pi = Type(6.283185307179586);
  Type sigma_min = Type(0.05); // minimum SD for lognormal likelihood to avoid overconfidence

  // -----------------------------
  // Bound penalties (soft, quadratic outside [lo, hi])
  // -----------------------------
  auto quad_penalty = [&](Type x, Type lo, Type hi, Type w){
    Type pen = Type(0);
    pen += CondExpLt(x, lo, w * (lo - x) * (lo - x), Type(0));
    pen += CondExpGt(x, hi, w * (x - hi) * (x - hi), Type(0));
    return pen;
  };
  Type pen = Type(0);
  Type w = Type(10.0); // penalty weight

  // Suggested biological ranges (used only for soft penalties)
  pen += quad_penalty(alpha_C, Type(0.0), Type(10.0), w);
  pen += quad_penalty(beta_C,  Type(0.0), Type(10.0), w);
  pen += quad_penalty(mu_C,    Type(0.0), Type(5.0),  w);
  pen += quad_penalty(q_C,     Type(0.0), Type(5.0),  w);

  pen += quad_penalty(mu_J,    Type(0.0), Type(5.0),  w);
  pen += quad_penalty(m_JA,    Type(0.0), Type(1.0),  w);
  pen += quad_penalty(J0,      Type(0.0), Type(5.0),  w);

  pen += quad_penalty(p_A,     Type(0.0), Type(1.0),  w);
  pen += quad_penalty(p_S,     Type(0.0), Type(1.0),  w);
  pen += quad_penalty(K_food,  Type(1.0), Type(150.0), w);
  pen += quad_penalty(eps_R,   Type(0.0), Type(1.0),  w);

  pen += quad_penalty(k_allee, Type(0.01), Type(50.0), w);
  pen += quad_penalty(C_allee, Type(0.0),  Type(5.0),  w);

  pen += quad_penalty(T_opt,   Type(20.0), Type(33.0), w);
  pen += quad_penalty(T_sd,    Type(0.1),  Type(5.0),  w);

  pen += quad_penalty(r_A,     Type(0.0),  Type(3.0),  w);
  pen += quad_penalty(r_S,     Type(0.0),  Type(1.0),  w);
  pen += quad_penalty(K_reef,  Type(50.0), Type(100.0), w);

  pen += quad_penalty(rec_A,   Type(0.0),  Type(50.0), w);
  pen += quad_penalty(rec_S,   Type(0.0),  Type(20.0), w);

  pen += quad_penalty(a_A,     Type(0.0),  Type(5.0),  w);
  pen += quad_penalty(a_S,     Type(0.0),  Type(5.0),  w);
  pen += quad_penalty(h_A,     Type(0.0),  Type(1.0),  w);
  pen += quad_penalty(h_S,     Type(0.0),  Type(1.0),  w);

  pen += quad_penalty(T_Aopt,  Type(20.0), Type(33.0), w);
  pen += quad_penalty(T_Asd,   Type(0.1),  Type(5.0),  w);
  pen += quad_penalty(T_Sopt,  Type(20.0), Type(33.0), w);
  pen += quad_penalty(T_Ssd,   Type(0.1),  Type(5.0),  w);
  pen += quad_penalty(T_bleach,Type(26.0), Type(33.0), w);
  pen += quad_penalty(k_bleach,Type(0.1),  Type(10.0), w);
  pen += quad_penalty(m_bleach_max, Type(0.0), Type(1.0), w);
  pen += quad_penalty(b_bleach_A,   Type(0.0), Type(2.0), w);
  pen += quad_penalty(b_bleach_S,   Type(0.0), Type(2.0), w);

  pen += quad_penalty(sigma_cots, Type(0.01), Type(2.0), w);
  pen += quad_penalty(sigma_fast, Type(0.01), Type(2.0), w);
  pen += quad_penalty(sigma_slow, Type(0.01), Type(2.0), w);

  // -----------------------------
  // State vectors for predictions
  // -----------------------------
  vector<Type> cots_pred(n); // predicted adult COTS density (ind m^-2)
  vector<Type> fast_pred(n); // predicted Acropora cover (%)
  vector<Type> slow_pred(n); // predicted slow-coral cover (%)
  vector<Type> juv_pred(n);  // predicted juvenile COTS density (ind m^-2)

  // Initialize with first observed values for observed states; juveniles from J0
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  juv_pred(0)  = J0;

  // -----------------------------
  // Time loop for state updates
  // -----------------------------
  for (int t = 1; t < n; t++) {
    // Previous states (t-1)
    Type C = cots_pred(t-1);   // previous adult COTS density (ind m^-2)
    Type J = juv_pred(t-1);    // previous juvenile COTS density (ind m^-2)
    Type A = fast_pred(t-1);   // previous Acropora cover (%)
    Type S = slow_pred(t-1);   // previous slow-coral cover (%)
    Type T = sst_dat(t-1);     // previous SST (deg C)
    Type I = cotsimm_dat(t-1); // previous immigration (ind m^-2 yr^-1)

    // 1) Multi-prey functional response denominator (saturating consumption) 
    Type D = Type(1.0) + h_A * a_A * A + h_S * a_S * S + eps; // add eps for stability

    // Per-capita predation mortality rates on corals (yr^-1)
    Type m_predA = C * a_A / D; // depends on COTS density, saturates with total prey
    Type m_predS = C * a_S / D; // slower due to preference if a_S < a_A

    // 2) Coral temperature modifiers and bleaching
    Type g_TA  = exp( -Type(0.5) * pow((T - T_Aopt) / (T_Asd + eps), 2) ); // Acropora temperature growth modifier
    Type g_TS  = exp( -Type(0.5) * pow((T - T_Sopt) / (T_Ssd + eps), 2) ); // Slow-coral temperature growth modifier
    Type b_ramp = Type(1.0) / (Type(1.0) + exp( -k_bleach * (T - T_bleach) )); // smooth bleaching onset above threshold
    Type m_bleachA = b_bleach_A * m_bleach_max * b_ramp; // Acropora bleaching mortality (yr^-1)
    Type m_bleachS = b_bleach_S * m_bleach_max * b_ramp; // Slow-coral bleaching mortality (yr^-1)

    // 3) Coral updates (multiplicative form ensures positivity)
    Type space_lim = (Type(1.0) - (A + S) / (K_reef + eps));                // free space fraction (dimensionless)
    Type A_next = A * exp( g_TA * r_A * space_lim - m_predA - m_bleachA )   // Acropora growth minus predation and bleaching
                  + rec_A * space_lim;                                      // density-independent recruitment to free space
    Type S_next = S * exp( g_TS * r_S * space_lim - m_predS - m_bleachS )   // Slow-coral update
                  + rec_S * space_lim;

    // 4) Food availability for COTS reproduction (preference-weighted, saturating)
    Type F = (p_A * A + p_S * S) / (K_food + A + S + eps);                  // dimensionless in [0, ~1]

    // 5) COTS temperature performance and Allee effect
    Type f_T = exp( -Type(0.5) * pow((T - T_opt) / (T_sd + eps), 2) );      // temperature suitability for recruitment
    Type g_allee = Type(1.0) / (Type(1.0) + exp( -k_allee * (C - C_allee) )); // smooth Allee effect [0,1]

    // 6) COTS recruitment to juveniles, juvenile survival/maturation, and adult survival
    Type R = alpha_C * C * exp( -beta_C * C ) * (eps_R * F) * f_T * g_allee // recruits to juvenile pool (ind m^-2 yr^-1)
             + I;                                                            // plus immigration (ind m^-2 yr^-1)

    Type J_survive_not_mature = (Type(1.0) - m_JA) * J * exp(-mu_J);        // juveniles surviving and not maturing
    Type J_next = J_survive_not_mature + R;                                  // next-year juveniles

    Type Srv_adult = exp( -mu_C - q_C * C );                                 // adult survival fraction (yr^-1)
    Type C_from_maturation = m_JA * J;                                       // juveniles maturing into adults
    Type C_next = C * Srv_adult + C_from_maturation;                         // next-year adults

    // Assign predictions (ensure minimal positivity for numerical stability in likelihood)
    cots_pred(t) = C_next + eps;
    fast_pred(t) = A_next + eps;
    slow_pred(t) = S_next + eps;
    juv_pred(t)  = J_next + eps;
  }

  // -----------------------------
  // Likelihood: Lognormal for strictly positive variables
  // -----------------------------
  Type nll = Type(0.0);

  Type sc = (sigma_cots > sigma_min ? sigma_cots : sigma_min); // effective sigma for COTS
  Type sf = (sigma_fast > sigma_min ? sigma_fast : sigma_min); // effective sigma for Acropora
  Type ss = (sigma_slow > sigma_min ? sigma_slow : sigma_min); // effective sigma for slow corals

  for (int t = 0; t < n; t++) {
    // COTS
    Type lc = log(cots_dat(t) + eps) - log(cots_pred(t) + eps);
    nll += Type(0.5) * (lc * lc / (sc * sc) + log(two_pi) + Type(2.0) * log(sc));

    // Acropora (fast)
    Type la = log(fast_dat(t) + eps) - log(fast_pred(t) + eps);
    nll += Type(0.5) * (la * la / (sf * sf) + log(two_pi) + Type(2.0) * log(sf));

    // Slow-coral
    Type ls = log(slow_dat(t) + eps) - log(slow_pred(t) + eps);
    nll += Type(0.5) * (ls * ls / (ss * ss) + log(two_pi) + Type(2.0) * log(ss));
  }

  // Add soft penalties
  nll += pen;

  // -----------------------------
  // Reporting
  // -----------------------------
  REPORT(cots_pred); // Predicted adult COTS density (ind m^-2)
  REPORT(fast_pred); // Predicted Acropora cover (%)
  REPORT(slow_pred); // Predicted slow-coral cover (%)
  REPORT(juv_pred);  // Predicted juvenile COTS density (ind m^-2)

  return nll;
}
