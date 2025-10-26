#include <TMB.hpp>
  
template <class Type> Type square(Type x){return x*x;}

template<class Type>
Type objective_function<Type>::operator() ()
{
  //===PULL IN DATA=============================================================
  DATA_INTEGER(first_yr); // First year 
  DATA_INTEGER(last_yr); // Last year 
  DATA_INTEGER(first_yr_dat); // First year 
  DATA_INTEGER(last_yr_dat); // Last year 
  DATA_INTEGER(NumYrs); // Number of years
  DATA_INTEGER(AgeClasses); // Number of CoTS age classes
  DATA_SCALAR(Kf);
  DATA_SCALAR(Km);
  DATA_SCALAR(rf);
  DATA_SCALAR(rm);
  DATA_SCALAR(p2f);
  DATA_SCALAR(p2m);
  DATA_SCALAR(P_init);
  DATA_SCALAR(B_init);
  DATA_SCALAR(ptil2);
  DATA_SCALAR(F_inv);
  DATA_SCALAR(p1CoTS);
  DATA_SCALAR(p2CoTS);
  DATA_VECTOR(phiCoTS);
  DATA_SCALAR(h);
  DATA_SCALAR(R0);
  DATA_SCALAR(rI);
  DATA_SCALAR(QI);
  DATA_SCALAR(KI);
  DATA_INTEGER(TP);
  DATA_SCALAR(Surv);
  DATA_SCALAR(RP);
  DATA_SCALAR(FP);
  DATA_SCALAR(Imm_CoTS);
  DATA_SCALAR(sigCoTS);
  DATA_IVECTOR(Imm_res_yrs); // Make sure values to be used in indexing are integer
  
  
  DATA_IVECTOR(Years);
  DATA_VECTOR(CoTS_dat);
  DATA_VECTOR(Cf_dat);
  DATA_VECTOR(Cm_dat);
  DATA_VECTOR(SST_dat);
  
  //====FITTED PARAMETERS=======================================================
  PARAMETER(dummy); // Dummy variable to test code runs
  PARAMETER(CoTS_init); // Starting CoTS numbers
  PARAMETER_VECTOR(immigration); // Immigration pulse in 1994
  PARAMETER(p1f); // CoTS impact on fast-growing coral
  PARAMETER(ptil); // Fast-growing coral impact on CoTS
  PARAMETER(p1m); // CoTS impact on slow-growing coral
  PARAMETER(Mcots); // Base CoTS instantaneous mortality
  PARAMETER(lam);
  PARAMETER(Cf_init);
  PARAMETER(Cm_init);
  PARAMETER(switchSlope);
  PARAMETER(Eta_f);
  PARAMETER(Eta_m);
  PARAMETER(M_SST50_f);
  PARAMETER(M_SST50_m);
  PARAMETER(Ble_imp_f);
  PARAMETER(Ble_imp_m);
  PARAMETER(SST0_f);
  PARAMETER(SST0_m);
  PARAMETER(SST_sig_f);
  PARAMETER(SST_sig_m);
  
  //====TEMP STRUCTURES=========================================================
  matrix<Type> N(NumYrs,AgeClasses); N.setZero(); // CoTS
  vector<Type> Cf(NumYrs); Cf.setZero(); // Fast-growing coral
  vector<Type> Cm(NumYrs); Cm.setZero(); // Slow-growing coral
  vector<Type> P(NumYrs); P.setZero(); // Fish predators
  vector<Type> B(NumYrs); B.setZero(); // Benthic invertebrates 
  vector<Type> Hcots(NumYrs); Hcots.setZero(); // Annual proportion of Age-2+ CoTS harvested
  vector<Type> Rcots(NumYrs); Rcots.setZero(); //Annual CoTS recruitment
  vector<Type> Yrs(NumYrs); Yrs.setZero(); // Simple vector to store years for plotting
  vector<Type> Imm_res(NumYrs); Imm_res.setZero(); // Vector for immigration residuals
  vector<Type> StockRec_res(NumYrs); StockRec_res.setZero(); // Vector for immigration residuals
  vector<Type> M_CoTS_age(AgeClasses); M_CoTS_age.setZero();
  
  //====TEMP VARIABLES==========================================================
  int yr,age;
  int index; //Loop indices 
  Type rho=0.0, f=0.0, g=0.0, g_TP=0.0, Qf=0.0, Qm=0.0, Qcots0=0.0, Qcots2=0.0; //Dynamic variables
  Type Kots_sp=0.0, SPR0=0.0, beta=0.0, alpha=0.0;
  Type rho_SST_F = 0.0, rho_SST_M = 0.0;
  Type M_ble_f = 0.0, M_ble_m = 0.0;
  
  //===FIRST YEAR INITIALISE=====================================================
  Cf(0) = Cf_init*(Km+Kf);//1.0*Kf; 
  Cm(0) = Cm_init*(Km+Kf);//1.0*Km; 
  
  P(0) = P_init; 
  B(0) = B_init; 
  for(age=0;age<AgeClasses;age++) {M_CoTS_age(age) = Mcots + lam/(1.0 + age);}
  
  N(0,0) = CoTS_init * exp(2*Mcots);//exp(2*M_CoTS_age(0));exp(M_CoTS_age(0)+M_CoTS_age(1));
  N(0,1) = CoTS_init *exp(1*Mcots);//exp(M_CoTS_age(1)); exp(M_CoTS_age(1));
  N(0,2) = CoTS_init;// * exp(0*Mcots);
  Yrs(0) = first_yr;
  
  // Assign fitted variables for immigration variability
  int size_immRes = Imm_res_yrs.size();
  for(index=0;index<size_immRes;index++) {Imm_res(Imm_res_yrs(index)-first_yr) = immigration(index);}
  
  
  //===COTS SPAWNING DERIVED VARIABLES==========================================
  Kots_sp = R0 * (exp(-2*Mcots)/(1.0 + exp(-Mcots))) +R0*exp(-Mcots);
  SPR0 = exp(-2*Mcots)/(1.0 + exp(-Mcots));
  beta = Kots_sp * ((1.0-h)/(5*h - 1.0));
  alpha = (beta + Kots_sp)/SPR0;
  
  
  //===EVOLVE POPULATIONS=======================================================
  for(yr=0;yr<NumYrs-1;yr++)
  { 
    //===INTERACTION FORMULATIONS===============================================
    
    rho = exp(-switchSlope*Cf(yr)/(Km+Kf));// Switch function
    f = (1.0 - ptil) +  ptil * rho; // Coral abundance on CoTS mortality
    
    Qf = Cf(yr)*(1.0-rho) *p1f*( ((N(yr,1)+N(yr,2))) / (1.0+exp(-(N(yr,1)+N(yr,2))/p2f))) ; // CoTS predation on Cf
    Qm = Cm(yr)*(rho) * p1m*((N(yr,1)+N(yr,2))/ (1.0+exp(-(N(yr,1)+N(yr,2))/p2m))); // CoTS predation on Cm
    
    Qcots0 = F_inv * N(yr,0) * B(yr); // Invertebrate predation on age-0 CoTS
    Qcots2 = (p1CoTS*N(yr,2)*P(yr)) / (1.0+exp(-N(yr,2)/p2CoTS)); // Fish predation on CoTS
    
    if(Qf>Cf(yr)) {Qf=Cf(yr)-0.001;} //Cf=0 is an absorbing state, avoid by defining small min amount 
    if(Qm>Cm(yr)) {Qm=Cm(yr)-0.001;} // As above 
    
    
    if(yr+first_yr<=last_yr_dat)
    {
      //SST on intrinsic growth rate
    rho_SST_F = exp( -pow((SST_dat(yr)-SST0_f),2)/(2*pow(SST_sig_f,2))  );
    //rho_SST_F = 1;
    rho_SST_M = exp( -pow((SST_dat(yr)-SST0_m),2)/(2*pow(SST_sig_m,2))  );
    //rho_SST_M = 1;
    
    // Bleaching mortality term
    
    M_ble_f = Cf(yr) * (1.0/( 1.0 + exp(-Eta_f*(SST_dat(yr)-M_SST50_f))));
    M_ble_m = Cm(yr) * (1.0/( 1.0 + exp(-Eta_m*(SST_dat(yr)-M_SST50_m))));
    //M_ble_f = Cf(yr) * Ble_imp_f;
    //M_ble_m = Cm(yr) * Ble_imp_m;
    if(M_ble_f>Cf(yr)) {M_ble_f=Cf(yr)-0.001;}
    if(M_ble_m>Cm(yr)) {M_ble_m=Cm(yr)-0.001;}
      
    }
    
    if(yr+first_yr>last_yr_dat)
    {
      //SST on intrinsic growth rate
      rho_SST_F = exp( -pow((27-SST0_f),2)/(2*pow(SST_sig_f,2))  );
      //rho_SST_F = 1;
      rho_SST_M = exp( -pow((27-SST0_m),2)/(2*pow(SST_sig_m,2))  );
      //rho_SST_M = 1;
      
      // Bleaching mortality term
      
      M_ble_f = Cf(yr) * (1.0/( 1.0 + exp(-Eta_f*(27-M_SST50_f))));
      M_ble_m = Cm(yr) * (1.0/( 1.0 + exp(-Eta_m*(27-M_SST50_m))));
      //M_ble_f = Cf(yr) * Ble_imp_f;
      //M_ble_m = Cm(yr) * Ble_imp_m;
      if(M_ble_f>Cf(yr)) {M_ble_f=Cf(yr)-0.001;}
      if(M_ble_m>Cm(yr)) {M_ble_m=Cm(yr)-0.001;} 
      
    }
    
    //===Population dynamics====================================================
    //---CoTS-------------------------------------------------------------------
    N(yr+1,1) = N(yr,0)*exp(-1*M_CoTS_age(0)); // Age-1
    N(yr+1,2) = N(yr,1)*exp(-f*M_CoTS_age(1)) + N(yr,2)*exp(-f*M_CoTS_age(2)); // Age-2+
    Rcots(yr+1) = (alpha * ((N(yr+1,2))/Kots_sp) ) / (beta + ((N(yr+1,2))/Kots_sp)); //Year-end recruitment allows us to pick up the N(yr+1,1)
    //N(yr+1,0) = Rcots(yr+1)* exp(StockRec_res(yr+1)) + Imm_CoTS * exp(Imm_res(yr+1)); // Age-0
    N(yr+1,0) = (Rcots(yr+1) + Imm_CoTS) * exp(Imm_res(yr+1) + pow(sigCoTS,2)/2); // Age-0
    
    
    //---Coral------------------------------------------------------------------
    Cf(yr+1) = Cf(yr)*(1.0 + rho_SST_F*rf*(1-(Cf(yr) + Cm(yr))/(Km+Kf)) ) - Qf - M_ble_f; // Fast-growing corals
    Cm(yr+1) = Cm(yr)*(1.0 + rho_SST_M*rm*(1-(Cf(yr) + Cm(yr))/(Km+Kf)) ) - Qm - M_ble_m; // Slow-growing corals

    
    Yrs(yr+1) = Yrs(yr)+1;
  }
  
  //=== COTS OBJ CONTRIBUTION ==================================================
  Type log_q=0.0, n_CoTS=CoTS_dat.size(), sig_CoTS=0.0, ObjFn_CoTS=0.0;
  Type n_Cf=Cf_dat.size(), n_Cm=Cm_dat.size(), sig_Cf=0.0, sig_Cm=0.0, ObjFn_Cf=0.0, ObjFn_Cm=0.0;
  Type srp=0.0;
  Type del=0.00;// // small number to stop logs blowing up (can use if neeeded)
  
  // CoTS catchability
  for(index=0;index<n_CoTS;index++)
  {log_q += (1.0/n_CoTS)*(log(CoTS_dat(index)+del) - log(N(Years(index)-first_yr,2)+del));}
  log_q=0.0;
  
  // CoTS std devs
  for(index=0;index<n_CoTS;index++)
  {sig_CoTS += square(log(CoTS_dat(index)+del) - log(exp(log_q)*N(Years(index)-first_yr,2)+del));}
  sig_CoTS = pow( ((1.0/n_CoTS)*sig_CoTS),0.5);//sqrt()
  
  // CoTS contribution to negative log likelihood function (objective function)
  for(index=0;index<n_CoTS;index++)
  {ObjFn_CoTS += (log(sig_CoTS+del) + square(log(CoTS_dat(index)+del) - log(exp(log_q)*N(Years(index)-first_yr,2)+del))/(2.0*square(sig_CoTS)));}
  
  //Penalty term
  for(index=0;index<size_immRes;index++)
  {srp += Imm_res(index)/(2*pow(sigCoTS,2));}
  
  
  
  //=== CORAL OBJ CONTRIBUTION ==================================================  
  Type cnvsn=(2/2); // Just here if adjusting as per bottom of table 2 in Morello et al 2014: Biom=COver^(3/2)
  
  for(index=0;index<n_Cf;index++)
  {sig_Cf += square(log( pow((Cf_dat(index)/100.0),(cnvsn)) +del) - log(Cf(Years(index)-first_yr)/(Km+Kf)+del));
    //std::cout<<"Years(index)-first_yr "<<Yrs(Years(index)-first_yr)<<"; Cf(Years(index)-first_yr)/Kf "<<Cf(Years(index)-first_yr)/Kf <<"; Cf_dat(index)/100.0 "<<Cf_dat(index)/100.0 <<"\n";
    }
  sig_Cf =pow(((1.0/n_Cf)*sig_Cf),0.5);//sqrt()
  
  for(index=0;index<n_Cf;index++)
  {ObjFn_Cf += (log(sig_Cf+del) + square(log(pow((Cf_dat(index)/100.0),(cnvsn))+del) - log(Cf(Years(index)-first_yr)/(Km+Kf)+del))/(2.0*square(sig_Cf)));}
  
  for(index=0;index<n_Cm;index++)
  {sig_Cm += square(log(pow((Cm_dat(index)/100.0),(cnvsn))+del) - log(Cm(Years(index)-first_yr)/(Km+Kf)+del));}
  sig_Cm = pow(((1.0/n_Cm)*sig_Cm),0.5);//sqrt()
  
  for(index=0;index<n_Cm;index++)
  {ObjFn_Cm += (log(sig_Cm+del) + square(log(pow((Cm_dat(index)/100.0),(cnvsn))+del) - log(Cm(Years(index)-first_yr)/(Km+Kf)+del))/(2.0*square(sig_Cm)));}
  /*
    jjjj
   */
  
  
  
  //===LIKELIHOOD CALCULATIONS==================================================
  Type obj_fun, NLL_vals;
  obj_fun = 0.0;
  NLL_vals = 0.0;
  
  //obj_fun = (dummy-0.5)*(dummy-0.5);
  obj_fun = ObjFn_Cf+ObjFn_Cm +ObjFn_CoTS;
  
  std::cout<<"--- obj_fun = "<<obj_fun<<"-------------------------------------"<<"\n";
  //std::cout<<"    ObjFn_CoTS="<<ObjFn_CoTS<<" srp="<<srp<<" ObjFn_Cf = "<<ObjFn_Cf<<" ObjFn_Cm = "<<ObjFn_Cm<<"\n";
  ADREPORT(obj_fun);
  ADREPORT(Cf);
  
  //===REPORT VARIABLES=========================================================
  REPORT(Yrs)
  REPORT(Cf);
  REPORT(Cm);
  REPORT(N);
  REPORT(obj_fun);
  //REPORT(NLL_vals)
  //REPORT(srp);
  //REPORT(log_q);
  return(obj_fun);
}

