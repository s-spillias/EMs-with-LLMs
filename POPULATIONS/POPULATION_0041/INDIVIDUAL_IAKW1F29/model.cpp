#include <TMB.hpp>

// Utility helpers
template<class Type> inline Type invlogit(Type x){ return Type(1)/(Type(1)+exp(-x)); }

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::pow;

  // -----------------------
  // 1) DATA (time-ordered)
  // -----------------------
  DATA_VECTOR(Year);        // Calendar year (used for indexing only)
  DATA_VECTOR(cots_dat);    // Adult COTS density [individuals m^-2]
  DATA_VECTOR(fast_dat);    // Fast coral (Acropora) cover [%]
  DATA_VECTOR(slow_dat);    // Slow coral (Faviidae/Porites) cover [%]
  DATA_VECTOR(sst_dat);     // Sea-surface temperature [°C]
  DATA_VECTOR(cotsimm_dat); // Larval immigration [individuals m^-2 yr^-1]

  int n = Year.size();
  // Safety: all series must match length
  // (ControlFile ensures merging; we still guard)
  n = CppAD::Integer( CppAD::CondExpLt(n, (int)cots_dat.size(), n, (int)cots_dat.size()) );
  n = CppAD::Integer( CppAD::CondExpLt(n, (int)fast_dat.size(), n, (int)fast_dat.size()) );
  n = CppAD::Integer( CppAD::CondExpLt(n, (int)slow_dat.size(), n, (int)slow_dat.size()) );
  n = CppAD::Integer( CppAD::CondExpLt(n, (int)sst_dat.size(),  n, (int)sst_dat.size()) );
  n = CppAD::Integer( CppAD::CondExpLt(n, (int)cotsimm_dat.size(), n, (int)cotsimm_dat.size()) );

  // -----------------------
  // 2) PARAMETERS
  // -----------------------
  // COTS population dynamics
  PARAMETER(log_r0);        // log intrinsic growth rate of adults (yr^-1); exp transform ensures positivity
  PARAMETER(gamma_food);    // exponent on food modulation (unitless, >0; saturating to amplify food-limited growth)
  PARAMETER(T_opt);         // SST of optimal COTS performance (°C)
  PARAMETER(log_sigma_temp);// log width of Gaussian thermal response (°C)
  PARAMETER(log_K_cots);    // log carrying capacity for adults (ind m^-2)
  PARAMETER(log_mC);        // log background adult mortality rate (yr^-1)
  PARAMETER(log_e_imm);     // log efficiency converting larval immigration to adults (unitless, yr^-1 equiv)

  // Predation/feeding on corals
  PARAMETER(log_a_feed);    // log attack/consumption rate (yr^-1 per ind m^-2)
  PARAMETER(log_h_half);    // log half-saturation for feeding (weighted % cover)
  PARAMETER(pref_fast);     // preference weight (fast); positive, normalized internally
  PARAMETER(pref_slow);     // preference weight (slow); positive, normalized internally

  // Coral regrowth and background mortality
  PARAMETER(log_gF);        // log intrinsic regrowth (fast) (yr^-1)
  PARAMETER(log_gS);        // log intrinsic regrowth (slow) (yr^-1)
  PARAMETER(mF0);           // background mortality (fast) (yr^-1, fraction of cover)
  PARAMETER(mS0);           // background mortality (slow) (yr^-1, fraction of cover)
  PARAMETER(log_h_recruit); // log half-saturation for COTS growth food modulation (weighted % cover)

  // Thermal bleaching mortality (smooth)
  PARAMETER(T_bleach);      // SST threshold for bleaching onset (°C)
  PARAMETER(bleach_slope);  // steepness of bleaching logistic (°C^-1)
  PARAMETER(bleach_max_F);  // max additional bleaching mortality (fast) (yr^-1)
  PARAMETER(bleach_max_S);  // max additional bleaching mortality (slow) (yr^-1)

  // Outbreak reinforcement (smooth Allee-like boost)
  PARAMETER(r_boost);       // additive boost to growth during outbreak (yr^-1)
  PARAMETER(C_thr);         // COTS density threshold for boost (ind m^-2)
  PARAMETER(log_c_slope);   // log scale of smooth threshold (ind m^-2)

  // Observation errors (log-scale where appropriate)
  PARAMETER(log_sigma_cots);// log obs SD for lognormal COTS
  PARAMETER(log_sigma_fast);// log obs SD for logit-normal fast coral
  PARAMETER(log_sigma_slow);// log obs SD for logit-normal slow coral

  // Initial states at first time step (t=1 in data)
  PARAMETER(init_C);        // initial adult COTS (ind m^-2)
  PARAMETER(init_F);        // initial fast coral cover (%)
  PARAMETER(init_S);        // initial slow coral cover (%)

  // -----------------------
  // 3) TRANSFORMS & CONST
  // -----------------------
  Type eps   = Type(1e-8);           // generic epsilon for positivity
  Type epsp  = Type(1e-6);           // slightly larger for proportions
  Type Kcov  = Type(100.0);          // total benthic cover capacity (%)
  Type r0    = exp(log_r0);          // intrinsic growth (yr^-1)
  Type sigmaT= exp(log_sigma_temp);  // thermal width (°C)
  Type Kcots = exp(log_K_cots);      // carrying capacity (ind m^-2)
  Type mC    = exp(log_mC);          // adult mortality (yr^-1)
  Type eImm  = exp(log_e_imm);       // conversion of immigrants to adults
  Type aFeed = exp(log_a_feed);      // predation rate
  Type hHalf = exp(log_h_half);      // feeding half-saturation
  Type hRec  = exp(log_h_recruit);   // food half-sat for growth
  Type cSlope= exp(log_c_slope);     // scale for outbreak threshold smoothness

  // Normalize coral preferences to sum to 1 to avoid identifiability
  Type pf = (pref_fast > eps ? pref_fast : eps);
  Type ps = (pref_slow > eps ? pref_slow : eps);
  Type wF = pf / (pf + ps + eps);    // fraction weight for fast coral
  Type wS = ps / (pf + ps + eps);    // fraction weight for slow coral

  // Observation SDs with floors for stability
  Type sd_min = Type(0.05); // minimum SD to avoid zero-variance issues
  Type sdC = exp(log_sigma_cots) + sd_min;
  Type sdF = exp(log_sigma_fast) + sd_min;
  Type sdS = exp(log_sigma_slow) + sd_min;

  // -----------------------
  // 4) STATE RECURSIONS
  // -----------------------
  vector<Type> cots_pred(n); // COTS prediction (ind m^-2)
  vector<Type> fast_pred(n); // Fast coral cover prediction (%)
  vector<Type> slow_pred(n); // Slow coral cover prediction (%)

  // Initialize previous states (t-1) using parameters
  Type C_prev = CppAD::CondExpGt(init_C, eps, init_C, eps);      // ensure positive
  Type F_prev = CppAD::CondExpGt(init_F, eps, init_F, eps);      // ensure >= eps
  Type S_prev = CppAD::CondExpGt(init_S, eps, init_S, eps);      // ensure >= eps
  // Ensure initial coral does not exceed capacity (soft cap via scaling)
  Type sumFS0 = F_prev + S_prev;
  if(sumFS0 > Kcov){
    F_prev = F_prev * (Kcov / (sumFS0 + eps));
    S_prev = S_prev * (Kcov / (sumFS0 + eps));
  }

  Type nll = Type(0.0);

  for(int t=0; t<n; t++){
    // (A) Modulators from forcings
    // Thermal modifier: Gaussian peak around T_opt
    Type therm = exp( -Type(0.5) * pow((sst_dat(t) - T_opt)/(sigmaT + eps), 2) ); // in (0,1]

    // Food availability for COTS growth (weighted coral cover, saturating)
    Type foodW = wF * F_prev + wS * S_prev;                       // weighted % cover
    Type fFood = foodW / (hRec + foodW + eps);                    // in (0,1)

    // Outbreak reinforcement (smooth threshold on C_prev)
    Type sOut  = invlogit( (C_prev - C_thr) / (cSlope + eps) );   // in (0,1)
    Type rEff  = r0 * therm * pow(fFood + eps, gamma_food)        // base-multiplicative growth
                 + r_boost * sOut;                                // outbreak persistence boost

    // Density regulation (Ricker form) + immigration input
    Type C_density = C_prev * exp( (rEff - mC) * (Type(1.0) - C_prev/(Kcots + eps)) );
    Type C_now     = C_density + eImm * cotsimm_dat(t);
    C_now = CppAD::CondExpGt(C_now, eps, C_now, eps);             // keep positive

    // (B) COTS predation pressure on corals (Holling II; bounded consumption)
    Type denom = hHalf + foodW + eps;

    // Per-guild instantaneous loss rates (yr^-1), scaled by C_prev and preference
    Type rF = aFeed * C_prev * (wF / denom);
    Type rS = aFeed * C_prev * (wS / denom);

    // Bounded losses over interval dt=1 yr using 1 - exp(-rate)
    Type predF = F_prev * (Type(1.0) - exp( -rF ));               // ≤ F_prev
    Type predS = S_prev * (Type(1.0) - exp( -rS ));               // ≤ S_prev

    // (C) Thermal bleaching mortality (smooth logistic onset)
    Type sBleach = invlogit( bleach_slope * (sst_dat(t) - T_bleach) ); // 0..1
    Type mortF_bleach = bleach_max_F * sBleach * F_prev;
    Type mortS_bleach = bleach_max_S * sBleach * S_prev;

    // (D) Coral regrowth with shared space limitation (logistic toward 100%)
    Type growthF = exp(log_gF) * F_prev * (Type(1.0) - (F_prev + S_prev)/Kcov);
    Type growthS = exp(log_gS) * S_prev * (Type(1.0) - (F_prev + S_prev)/Kcov);

    // (E) Background mortality
    Type mortF0 = mF0 * F_prev;
    Type mortS0 = mS0 * S_prev;

    // (F) Update coral states (smooth bounded dynamics; apply soft floors and soft caps)
    Type F_now = F_prev + growthF - mortF0 - mortF_bleach - predF;
    Type S_now = S_prev + growthS - mortS0 - mortS_bleach - predS;

    // Soft floors at ~0 (avoid negative)
    F_now = CppAD::CondExpGt(F_now, eps, F_now, eps);
    S_now = CppAD::CondExpGt(S_now, eps, S_now, eps);

    // Soft cap total cover via proportional rescaling if over capacity
    Type sumFS = F_now + S_now;
    if(sumFS > Kcov){
      F_now = F_now * (Kcov / (sumFS + eps));
      S_now = S_now * (Kcov / (sumFS + eps));
    }

    // (G) Record predictions for time t (computed only from states at t-1 and forcings at t)
    cots_pred(t) = C_now;
    fast_pred(t) = F_now;
    slow_pred(t) = S_now;

    // (H) Likelihood contributions (all observations included if finite)

    // COTS: lognormal on strictly positive scale
    if( R_IsNA(asDouble(cots_dat(t))) == 0 ){
      Type y  = cots_dat(t);
      Type mu = log(cots_pred(t) + eps);
      Type z  = log(y + eps);
      nll -= dnorm(z, mu, sdC, true);
    }

    // Corals: logit-normal on [0,1] scale (after scaling from %), with stability floors
    if( R_IsNA(asDouble(fast_dat(t))) == 0 ){
      Type y  = fast_dat(t) / Type(100.0);
      y = CppAD::CondExpGt(y, epsp, y, epsp);
      y = CppAD::CondExpLt(y, Type(1.0)-epsp, y, Type(1.0)-epsp);
      Type p  = fast_pred(t) / Type(100.0);
      p = CppAD::CondExpGt(p, epsp, p, epsp);
      p = CppAD::CondExpLt(p, Type(1.0)-epsp, p, Type(1.0)-epsp);
      Type mu = log(p/(Type(1.0)-p));
      Type z  = log(y/(Type(1.0)-y));
      nll -= dnorm(z, mu, sdF, true);
    }

    if( R_IsNA(asDouble(slow_dat(t))) == 0 ){
      Type y  = slow_dat(t) / Type(100.0);
      y = CppAD::CondExpGt(y, epsp, y, epsp);
      y = CppAD::CondExpLt(y, Type(1.0)-epsp, y, Type(1.0)-epsp);
      Type p  = slow_pred(t) / Type(100.0);
      p = CppAD::CondExpGt(p, epsp, p, epsp);
      p = CppAD::CondExpLt(p, Type(1.0)-epsp, p, Type(1.0)-epsp);
      Type mu = log(p/(Type(1.0)-p));
      Type z  = log(y/(Type(1.0)-y));
      nll -= dnorm(z, mu, sdS, true);
    }

    // Advance states
    C_prev = C_now;
    F_prev = F_now;
    S_prev = S_now;
  }

  // -----------------------
  // 5) REPORTING
  // -----------------------
  // Predictions corresponding to observed series (suffix _pred)
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Helpful diagnostics
  REPORT(wF);                // normalized feeding preference (fast)
  REPORT(wS);                // normalized feeding preference (slow)
  REPORT(Kcots);             // carrying capacity (adults)
  REPORT(r0);                // intrinsic growth (yr^-1)
  REPORT(aFeed);             // attack rate (yr^-1 per ind m^-2)

  // Return the total negative log-likelihood
  return nll;
}

/*
Model equation annotations (per time step t; states evaluated from previous states):
1) Thermal modifier: therm = exp(-0.5 * ((SST_t - T_opt)/sigma_T)^2)
2) Food saturation: fFood = (wF*F_{t-1} + wS*S_{t-1}) / (h_rec + wF*F_{t-1} + wS*S_{t-1})
3) Outbreak smooth boost: sOut = invlogit((C_{t-1} - C_thr) / c_slope)
4) Effective growth: rEff = r0 * therm * fFood^{gamma_food} + r_boost * sOut
5) COTS density regulation: C_t = C_{t-1} * exp((rEff - mC) * (1 - C_{t-1}/K_cots)) + eImm * cotsimm_t
6) Feeding denominators: denom = h_half + wF*F_{t-1} + wS*S_{t-1}
7) Coral predation losses: predF = F_{t-1} * (1 - exp(-aFeed * C_{t-1} * wF / denom)), similarly for predS
8) Bleaching: sBleach = invlogit(bleach_slope * (SST_t - T_bleach)), then
   mortF_bleach = bleach_max_F * sBleach * F_{t-1}, mortS_bleach analogously
9) Coral growth (logistic toward 100% space):
   F_t = F_{t-1} + gF*F_{t-1}*(1 - (F_{t-1}+S_{t-1})/100) - mF0*F_{t-1} - mortF_bleach - predF
   S_t = S_{t-1} + gS*S_{t-1}*(1 - (F_{t-1}+S_{t-1})/100) - mS0*S_{t-1} - mortS_bleach - predS
10) Observation models:
   - COTS: lognormal on C_t with sdC
   - Corals: logit-normal on F_t/100 and S_t/100 with sdF and sdS
Notes:
- All divisions have eps added for numerical safety.
- Predictions cots_pred, fast_pred, slow_pred are the states C_t, F_t, S_t computed from (t-1) states and time-t forcings only (no data leakage).
*/
