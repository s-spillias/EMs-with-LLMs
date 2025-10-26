rm(list=ls()) # clear everything before starting 
# setwd("C:\\Users\\rog151\\OneDrive - CSIRO\\Documents\\MICE training\\Simplified_Code\\Master_clean")

require(TMB)
compile('CoTSmodel_v3.cpp') # Name of file here and in next line need to match and match the call to the model further down
dyn.load(dynlib('CoTSmodel_v3'))
#dev.off(dev.list()["RStudioGD"]) # will thow an error if no current figures to delete

# ==== DATA ====================================================================
first_yr <- 2000 # first "simulation" year
last_yr <- 2025 # last "simulation" year
first_yr_dat <- 2000 # first year of data
last_yr_dat <- 2025 # last year of data
NumYrs <- length(seq(first_yr,last_yr,by=1)) # Number of years in data
AgeClasses <-3 # Number of CoTS age classes (inc recruit class)
K <- 3000.0; # Coral carrying capacity (joint carrying capacity)
rf <- 0.5 # Intrinsic growth rate for branching fast-growing corals
rm <-  0.1 # Intrinsic growth rate for massive slow-growing corals
p2f <- 10.0 # Fast-growing predation density-dependence decay constant (shape param that controls rate at which coral consumption tends to linear, is "exponential" shaped at low density)
p2m <- 8.0 # Slow-growing predation Density-dependence decay constant (shape param that controls rate at which coral consumption tends to linear, is "exponential" shaped at low density)
h <- 0.5 # Slope of stock recruitment curve at 20 % of unexploited spawning biomass
R0 <- 1.0 # Recruitment in unexploited population (this is used relatively and having =1 simplifies things)
Imm_CoTS <- 1.0 # Background base immigration rate
sigCoTS <- 0.7 # Assumed variability in CoTS recruitment (about the stock-recruitment curve)
Imm_res_yrs <- c(2009,2011,2012) # Year in which there is a pulse of CoTS recuits (immigration or self-recruitment)

# --- Could try to fit these, but toy example and keeping it simple ------------
# Have left in the params to be fitted denoted by "###" so that all explanations
# included here for reference
CoTS_init <- 0.3 # Initialised CoTS
### immigration <- c(1.5,1.6,0.7) ### Recruitment deviations 
p1f <- 0.15 # Mortality of fast-growing corals induced by a CoTS (sensu consumption rate)
ptil <- 0.5 # Max component of overall CoTS mortality (inc age-dependence) attributable to fast-growing coral availability  
p1m <- 0.06 # Mortality of slow-growing corals induced by a CoTS (sensu consumption rate)
### Mcots <- 2.3 # Instantaneous CoTS mortality 
lam <- 0.0 # Controls component of base CoTS mortality attributable to age dependence
Cf_init <- 0.16 # Initialised branching fast-growing corals
Cm_init <- 0.12 # Initialised massive slow-growing corals
switchSlope <- 5 # Shape parameter that control slope of prey switching function between fast- and slow-goring corals
### Eta_f <- 2.0 # Slope of bleaching mortality curve for fast-growing corals 
### Eta_m <- 1.0 # Slope of bleaching mortality curve for slow-growing corals 
### M_SST50_f <- 31 # SST at which observe 50 % mortality of fast-growing corals 
### M_SST50_m <- 32 # SST at which observe 50 % mortality of slow-growing corals 
### Ble_imp_f <- 0.0 # Option for impulse bleaching mortality for branching fast-growing corals
### Ble_imp_m <- 0.0 # Option for impulse bleaching mortality for massive slow-growing corals
SST0_f <- 26 # Optimal growth SST for branching fast-growing corals 
SST0_m <- 27 # Optimal growth SST for massive slow-growing corals
SST_sig_f <- 2.0 # Variability of branching fast-growing corals growth about optimum (Gaussian shaped)
SST_sig_m <- 4.0 # Variability of massive slow-growing corals growth about optimum (Gaussian shaped)

# --- Pull in the data from CSV file -----------------------------------
dataD <- read.csv("timeseries_data.csv")
Years <- dataD$Years # Years for observations
CoTS_dat <- dataD$CoTS_dat # CoTS observation in given year
Cf_dat <- dataD$Cf_dat # fast-growing coral observation in given year
Cm_dat <- dataD$Cm_dat # slow-growing coral observation in given year
SST_dat <- dataD$SST_dat # Sea Surface temperature observation in a given year

# ==== MODEL PHASING ===========================================================
# Just fit a dummy variable:
map0 <- list(immigration=factor(c(NA,NA,NA)), Mcots=factor(NA), Eta_f=factor(NA), 
             Eta_m=factor(NA), M_SST50_f=factor(NA), M_SST50_m=factor(NA),
             Ble_imp_f=factor(NA),Ble_imp_m=factor(NA))

# Fit MCoTS:
map1 <- list(dummy=factor(NA), immigration=factor(c(NA,NA,NA)), Eta_f=factor(NA), 
             Eta_m=factor(NA), M_SST50_f=factor(NA), M_SST50_m=factor(NA),
             Ble_imp_f=factor(NA),Ble_imp_m=factor(NA))

# Fit MCoTS and M_SST50_f:
map2 <- list(dummy=factor(NA), immigration=factor(c(NA,NA,NA)), Eta_f=factor(NA), 
             Eta_m=factor(NA), M_SST50_m=factor(NA), Ble_imp_f=factor(NA),
             Ble_imp_m=factor(NA))

# Fit MCoTS, M_SST50_f, and Eta_f=factor:
map3 <- list(dummy=factor(NA), immigration=factor(c(NA,NA,NA)),Eta_f=factor(NA),  
             Eta_m=factor(NA), Ble_imp_f=factor(NA),
             Ble_imp_m=factor(NA))

# Optional bounds for map2:
# have to list for params in order, here its (McoTS, M_SST50_f)
LBs_map2 <- c(-Inf,20); # Lower bounds
UBs_map2 <- c(Inf, 40); # Upper bounds

LBs_map3 <- c(-Inf,20,-Inf); # Lower bounds
UBs_map3 <- c(Inf, 40,Inf); # Upper bounds

# ==== MODEL WORKFLOW ==========================================================
data <- list(first_yr = first_yr,last_yr = last_yr,first_yr_dat = first_yr_dat,
             last_yr_dat = last_yr_dat, NumYrs = NumYrs, AgeClasses=AgeClasses, 
             K=K, rf=rf, rm=rm, p2f=p2f, p2m=p2m, h=h, R0=R0,Imm_CoTS=Imm_CoTS,
             sigCoTS=sigCoTS,Imm_res_yrs=Imm_res_yrs, Years=Years, CoTS_dat=CoTS_dat, 
             Cf_dat=Cf_dat, Cm_dat=Cm_dat,SST_dat=SST_dat, CoTS_init=CoTS_init, 
             p1f=p1f, ptil=ptil, p1m=p1m, lam=lam, Cf_init=Cf_init, Cm_init=Cm_init, 
             switchSlope=switchSlope, SST0_f=SST0_f, SST0_m=SST0_m, SST_sig_f=SST_sig_f, 
             SST_sig_m=SST_sig_m)
#Mcots=2.3, M_SST50_f=31
# --- PHASE 1 ------------------------------------------------------------------
parameters <- list(dummy=5.0, immigration=c(1.5,1.6,0.7), Mcots=2.5, Eta_f=2.0, 
                   Eta_m=1.0, M_SST50_f=34, M_SST50_m=32,Ble_imp_f=0.0, 
                   Ble_imp_m=0.0)
model <- MakeADFun(data, parameters, DLL="CoTSmodel_v3",silent=T,map=map2)
fit <- nlminb(model$par, model$fn, model$gr)#,lower=LBs_map3,upper=UBs_map3)
best1 <- model$env$last.par.best

# --- PHASE 2 ------------------------------------------------------------------
parameters <- model$env$parList(fit$par)#grabs all out params from phase 1 and then passes to next phase as updated list
model <- MakeADFun(data, parameters, DLL="CoTSmodel_v3",silent=T,map=map3)
fit <- nlminb(model$par, model$fn, model$gr,lower=LBs_map3,upper=UBs_map3)
best2 <- model$env$last.par.best

# === EXTRACT FROM REPORT ============================================
Coral_f <- model$report()$Cf
Coral_m <- model$report()$Cm
CoTS <- model$report()$N
years <- model$report()$Yrs
q <- exp(model$report()$log_q)
rep <- sdreport(model)

# === PLOT OUTPUTS =============================================================
# --- Plot SST -----------------------------------------------------------------
plot(SST_dat,type="l")

# --- Plot fast-growing corals--------------------------------------------------
plot(years,100*((Coral_f/(K))), type = "l", xaxt='n',ylim=c(0,100), xlim=c(first_yr, last_yr), yaxt='n'); 
axis(side=2, at=seq(0, 100, by=0.5)); 
axis(side=1, at=seq(first_yr, last_yr, by=4));
points(seq(first_yr, last_yr,by=1),Cf_dat)

# --- Plot slow-growing corals -------------------------------------------------
plot(years,100*((Coral_m/(K))), type = "l", xaxt='n',ylim=c(0,100), xlim=c(first_yr, last_yr), yaxt='n'); 
axis(side=2, at=seq(0, 100, by=2)); 
axis(side=1, at=seq(first_yr, last_yr, by=4));
points(seq(first_yr, last_yr,by=1),Cm_dat)

# --- Plot age-2+ CoTS ---------------------------------------------------------
plot(years,1*CoTS[,3], type = "l", xaxt='n',ylim=c(0,5), xlim=c(first_yr,last_yr), yaxt='n'); 
axis(side=2, at=seq(0, 5, by=0.5)); 
axis(side=1, at=seq(first_yr, last_yr, by=2)); 
points(seq(first_yr, last_yr,by=1),CoTS_dat)

# === LIKELIHOOD PROFILES ======================================================
# Ideally check each parameter - loop through them to make sure no bad ones
cat("---------------------------------------------------\n")
if(rep$pdHess) # Only run if Hessian is positive definite 
{
  param_name <- attributes(rep$par)$names
  
  for(ii in 1:length(attributes(rep$par)$names))
  {
    prof <- tmbprofile(obj=model,name=param_name[ii],trace = FALSE) # Compute likelihood profile (trace true/false will output vals or not to console)
    CI<-confint(prof) # Get profile-based confidence interval (default 95% CI)
    
    # Plot the profile for "param_name":
    plot(prof[,1], prof$value-min(prof$value), type = "l",main=param_name[ii], 
         xlab="parameter value",ylab="Maximum - marginal neg likelihood")
    
    # Plot estimated value:
    abline(v=as.numeric(model$env$parList(fit$par)[param_name[ii]]),col="red") 
    
    # Plot 95 % CI of value:
    abline(h=1.92,col="blue") # Intersection points based on test statistic being asymptotically chi-squared distributed with df=1
    abline(v=CI,col="blue") # Vertical bars where parameter CI is
    
    cat(param_name[ii],": 95 % prof CI [", round(CI,4), " ]\n")
  }
}
cat("\n---------------------------------------------------\n")
# === ASYMPTOTIC CI  ===========================================================
if(rep$pdHess) # Only run if Hessian is positive definite 
{
  confidence_level <- 0.95
  
  for(ii in 1:length(attributes(rep$par)$names))
  {
    mean_value <- as.numeric(model$env$parList(fit$par)[param_name[ii]])
    standard_error <- as.numeric(summary(sdreport(model))[param_name[ii],"Std. Error"])
    critical_value <- qnorm((1 + confidence_level) / 2) 
    # Calculate confidence interval
    lower_bound <- mean_value - critical_value * standard_error
    upper_bound <- mean_value + critical_value * standard_error
    # Print the result
    cat(param_name[ii],":",confidence_level*100,"% asym CI [", round(lower_bound, 4), 
        round(upper_bound, 4),"], CV=",abs(standard_error/mean_value) ,"\n")
  }
  
  CV_CritVal <- 1.0
  cat("\nShortlist of parameters with high CVs (>", CV_CritVal,"):\n")
  for(ii in 1:length(attributes(rep$par)$names))
  {
    if(abs(standard_error/mean_value)>CV_CritVal)
    {
      cat("* ",param_name[ii],":",abs(standard_error/mean_value),"\n")
    }
  }
  
}
cat("\n---------------------------------------------------\n")
# === COMPUTE AIC ==============================================================
# Can do multiple models in here if wanting
if(rep$pdHess) # Only run if Hessian is positive definite 
{
  numParams <- length(attributes(rep$par)$names)
  nLL <- as.numeric(rep$value["obj_fun"])
  AIC_calc <- 2*numParams - 2*(-nLL)
  cat("AIC:",AIC_calc,"\n")
}
cat("---------------------------------------------------\n")
# === EXAMINE CORRELATIONS =====================================================
if(rep$pdHess) # Only run if Hessian is positive definite 
{
  correlations <- as.data.frame(cov2cor(solve(model$he())),row.names=attributes(rep$par)$names,col.names=attributes(rep$par)$names)
  colnames(correlations) <- attributes(rep$par)$names
  rownames(correlations) <- attributes(rep$par)$names
  print(correlations)
}

corCritVal <- 0.85
cat("\nCorrelated Parameters (>", corCritVal,"):\n")
for(index in 1:length(correlations)) 
{
  # Matrix always square so exclude diagonal (will be 1 as of course perfectly 
  # correlated with itself).
  correlatedParams <- colnames(correlations[,-index])[abs(correlations[index,-index]) >= corCritVal] 
  if(length(correlatedParams)>0) {cat("* ",rownames(correlations)[index],":",correlatedParams,"\n")}
}

# === CHECK IF PARAMETERS CLOSE TO BOUNDS (IF APPLICABLE) ======================
if(rep$pdHess) # Only run if Hessian is positive definite 
{
  
  
  
  
}

cat("\n---------------------------------------------------\n")
# === CHECK FINAL GRADIENT =====================================================
if(rep$pdHess) # Only run if Hessian is positive definite 
{
  tolerance <- 1.0e-04
  # Final maximal gradient component
  gradOut <- max(abs(rep$gradient.fixed))
  if(gradOut > tolerance)
  {
    cat("\nMax gradient component > tolerance: ",gradOut,">",tolerance,"\n")
  }
  if(gradOut <= tolerance)
  {
    cat("\nMax gradient component <= tolerance: ",gradOut,"<=",tolerance,"\n")
  }
}

cat("\n---------------------------------------------------\n")
# === PRINT OUT REPORT =========================================================
cat("\nq=",q,"\n\n")
print(rep)
cat("---------------------------------------------------\n")
