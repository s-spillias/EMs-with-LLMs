#include <TMB.hpp>

// Helper functions for numerical stability
// Use TMB-provided invlogit; do not redefine to avoid conflicts.
// Stable softplus without using std::log1p (works with AD types)
template<class Type>
Type softplus(Type x){
  Type zero = Type(0);
  // softplus(x) = max(0,x) + log(1 + exp(-|x|)) implemented with AD-safe conditionals
  return CppAD::CondExpGt(
    x, zero,
    x + log(Type(1) + exp(-x)),
    log(Type(1) + exp(x))
  );
}

/* 
Model overview and equations (all time indices t refer to rows in the data; Year is the time key):
State variables:
- A_t: Adult COTS density (individuals m^-2)
- F_t: Fast coral (Acropora) cover fraction (0-1); fast_dat are percentages, so fast_pred = 100*F_t
- S_t: Slow coral (Faviidae/Porites) cover fraction (0-1); slow_pred = 100*S_t
- H_t = 1 - F_t - S_t: Free space fraction (can be <0 transiently; growth terms use saturating H_t/(H_t + K_space) to ensure smooth behavior)

Forcings (observed):
- sst_dat[t