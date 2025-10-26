rm(list=ls())
setwd("E:\\CSIRO_laptop_Jan_2025_copy\\Docs\\MICE training\\SentCode\\checkRun")

require(TMB)
compile('CoTSmodel_v3.cpp')
dyn.load(dynlib('CoTSmodel_v3'))
#dev.off(dev.list()["RStudioGD"])# will thow an error if no current figures to delete

# ==== DATA ====================================================================
first_yr <- 2000
last_yr <- 2025
first_yr_dat <- 2000
last_yr_dat <- 2025
NumYrs <- length(seq(first_yr,last_yr,by=1))
AgeClasses <-3
Kf <- 2500.0
Km <- 500.0
rf <- 0.5 
rm <-  0.1
p2f <- 10.0
p2m <- 8.0
P_init <- 10.0
B_init <- 0.0
ptil2 <- 0.0
F_inv <- 0.0
p1CoTS <- 0.0
p2CoTS <- 50.0
phiCoTS <- c(0,1)
h <- 0.5
R0 <- 1.0
rI <- 1.5
QI <- 0.0
KI <- 1.0
TP <- 4
Surv <- 0.8
RP <- 0.3905
FP <- 0.0
Imm_CoTS <- 1.0
sigCoTS <- 0.7
Imm_res_yrs <- c(2009,2011,2012) #94 Has to be same size as residuals vector


NumRow <- scan("MorelloModel_data.dat",skip=1,n=1)
NumCol <- scan("MorelloModel_data.dat",skip=2,n=1)
dataD <- as.matrix(read.table("MorelloModel_data.dat",skip=3,header=FALSE,nrow=NumRow))
Years <- dataD[,1]
CoTS_dat <- dataD[,2]
Cf_dat <- (dataD[,3])
Cm_dat <- (dataD[,4])
SST_dat<- (dataD[,5]) 
plot(SST_dat,type="l")

# ==== MODEL PHASING ===========================================================
map0 <- list(CoTS_init=factor(NA),
             immigration=factor(c(NA,NA,NA)),
             p1f=factor(NA),ptil=factor(NA),p1m=factor(NA),Mcots=factor(NA),lam=factor(NA),
             Cf_init=factor(NA),Cm_init=factor(NA),switchSlope=factor(NA),Eta_f=factor(NA), 
             Eta_m=factor(NA), M_SST50_f=factor(NA), M_SST50_m=factor(NA),Ble_imp_f=factor(NA), Ble_imp_m=factor(NA),
             SST0_f=factor(NA), SST0_m=factor(NA), SST_sig_f=factor(NA), SST_sig_m=factor(NA))

map0b <- list(dummy=factor(NA),CoTS_init=factor(NA),
             immigration=factor(c(NA,NA,NA)),
             p1f=factor(NA),ptil=factor(NA),p1m=factor(NA),lam=factor(NA),
             Cf_init=factor(NA),Cm_init=factor(NA),switchSlope=factor(NA),Eta_f=factor(NA), 
             Eta_m=factor(NA), M_SST50_f=factor(NA), M_SST50_m=factor(NA),Ble_imp_f=factor(NA), Ble_imp_m=factor(NA),
             SST0_f=factor(NA), SST0_m=factor(NA), SST_sig_f=factor(NA), SST_sig_m=factor(NA))

map0c <- list(dummy=factor(NA),CoTS_init=factor(NA),
              immigration=factor(c(NA,NA,NA)),
              p1f=factor(NA),ptil=factor(NA),p1m=factor(NA),lam=factor(NA),
              Cf_init=factor(NA),Cm_init=factor(NA),switchSlope=factor(NA),Eta_f=factor(NA), 
              Eta_m=factor(NA),Ble_imp_f=factor(NA), Ble_imp_m=factor(NA),
              SST0_f=factor(NA), SST0_m=factor(NA), SST_sig_f=factor(NA), SST_sig_m=factor(NA))

map1 <- list(dummy=factor(NA), CoTS_init=factor(NA),
             immigration=factor(c(NA)),lam=factor(NA),Cf_init=factor(NA),Cm_init=factor(NA),switchSlope=factor(NA))
map2 <- list(dummy=factor(NA),p1f=factor(NA),ptil=factor(NA),p1m=factor(NA),Mcots=factor(NA),lam=factor(NA),Cf_init=factor(NA),Cm_init=factor(NA),switchSlope=factor(NA))
map3 <- list(dummy=factor(NA),lam=factor(NA),Cf_init=factor(NA),Cm_init=factor(NA),switchSlope=factor(NA))

LBs_1 <-c(0.001,0.001,0.001,0.5) # p1f,ptil,p1m,Mcots
UBs_1 <-c(1.0,1.0,1.0,4.0)

LBs_2 <-c(0.001,0,0) # Cinit, res, imm
UBs_2 <-c(1.00,Inf,Inf)

LBs_3 <-c(0.001,0,0,0.001,0.001,0.001,0.5) # Cinit, res, imm, p1f,ptil,p1m,Mcots
UBs_3 <-c(1.00,Inf,Inf,1.0,1.0,1.0,5.0)
  
# ==== MODEL WORKFLOW ==========================================================
#Morello fitted vals: CoTS_init=0.505,residuals=4.307,immigration=4.292,p1f=0.129,ptil=0.258,p1m=0.268,Mcots=2.560,lam=0.00
data <- list(first_yr = first_yr,last_yr = last_yr,first_yr_dat = first_yr_dat,last_yr_dat = last_yr_dat, NumYrs = NumYrs,
             AgeClasses=AgeClasses, Kf=Kf, Km=Km, rf=rf, rm=rm, p2f=p2f, p2m=p2m,
             P_init=P_init, B_init=B_init, ptil2=ptil2, F_inv=F_inv, p1CoTS=p1CoTS,
             p2CoTS=p2CoTS, phiCoTS=phiCoTS, h=h, R0=R0, rI=rI, QI=QI, KI=KI,
             TP=TP, Surv=Surv, RP=RP, FP=FP, Imm_CoTS=Imm_CoTS,sigCoTS=sigCoTS, 
             Imm_res_yrs=Imm_res_yrs,#StockRec_res_yrs=StockRec_res_yrs, 
             Years=Years,
             CoTS_dat=CoTS_dat, Cf_dat=Cf_dat, Cm_dat=Cm_dat,SST_dat=SST_dat)

# --- PHASE 1 ------------------------------------------------------------------
parameters <- list(dummy=5.0,CoTS_init=0.3,
                   immigration=c(1.5,1.6,0.7),p1f=0.15,ptil=0.5,p1m=0.06,
                   Mcots=2.1,lam=0.0,Cf_init=0.16,Cm_init=0.12, switchSlope=5,
                   Eta_f=2.0, Eta_m=1.0, M_SST50_f=28, M_SST50_m=35,Ble_imp_f=0.0, 
                   Ble_imp_m=0.0, SST0_f=26, SST0_m=27, SST_sig_f=2.0, SST_sig_m=4.0)
# Mcots=2.3
# M_SST50_f=31
# M_SST50_m=32
model <- MakeADFun(data, parameters, DLL="CoTSmodel_v3",silent=T,map=map0b)
fit <- nlminb(model$par, model$fn, model$gr) #,lower=LBs_1,upper=UBs_1
best1 <- model$env$last.par.best


# --- PHASE 2 ------------------------------------------------------------------
# fitting 3 params now (SST50 for M and F, also Mcots)
parameters2 <- model$env$parList(fit$par) #grab parameters with updated values

LBs <-c(1.0,25.0,26.0) # Mcots, SST50_F, SST50_M
UBs <-c(3.0,33.0,34.0)

model <- MakeADFun(data, parameters2, DLL="CoTSmodel_v3",silent=T,map=map0c)
fit <- nlminb(model$par, model$fn, model$gr,lower=LBs,upper=UBs) #including upper and lower bounds
best2 <- model$env$last.par.best



# === EXTRACT VARIABLES FROM REPORT ============================================
Coral_f <- model$report()$Cf
Coral_m <- model$report()$Cm
CoTS <- model$report()$N
OBJ <- model$report()$obj_fun

years <- model$report()$Yrs
#q <- exp(model$report()$log_q)

Cf_points <- (100*Coral_f/(Km+Kf))*exp(rnorm(length(Coral_f),mean=0,sd=0.27))
plot(years,100*((Coral_f/(Km+Kf))), type = "l", xaxt='n',ylim=c(0,100), xlim=c(first_yr, last_yr), yaxt='n'); axis(side=2, at=seq(0, 100, by=0.5)); axis(side=1, at=seq(first_yr, last_yr, by=4));points(seq(first_yr, last_yr,by=1),Cf_dat)

Cm_points <- (100*Coral_m/(Km+Kf))*exp(rnorm(length(Coral_m),mean=0,sd=0.3))
plot(years,100*((Coral_m/(Km+Kf))), type = "l", xaxt='n',ylim=c(0,100), xlim=c(first_yr, last_yr), yaxt='n'); axis(side=2, at=seq(0, 100, by=2)); axis(side=1, at=seq(first_yr, last_yr, by=4));points(seq(first_yr, last_yr,by=1),Cm_dat)


CoTS_points <- CoTS[,3]*exp(rnorm(length(CoTS[,3]),mean=0,sd=0.3))
plot(years,1*CoTS[,3], type = "l", xaxt='n',ylim=c(0,5), xlim=c(first_yr,last_yr), yaxt='n'); axis(side=2, at=seq(0, 5, by=0.5)); axis(side=1, at=seq(first_yr, last_yr, by=2)); points(seq(first_yr, last_yr,by=1),CoTS_dat)

prof <- tmbprofile(obj=model,name="M_SST50_m",trace=FALSE)
plot(prof$M_SST50_m ,prof$value-min(prof$value), type = "l")
abline(h=1.92,col="blue") #95% CI 
abline(v=as.numeric(best2[3]),col="red") #Mcots est
CI<-confint(prof) #returns confidence interval (default 95% CI)
abline(v=CI,col="blue")

numParams <- 1
AIC <- -2*log(exp(-OBJ))+2*numParams
print(AIC)

rep <- summary(sdreport(model))
Cfrep <- rep[rownames(rep) == "Cf", ] # get Cf with standard errors
Cf_est <- matrix(Cfrep[, "Estimate"], ncol = 1) #extract estimates
Cf_stdEr <- matrix(Cfrep[, "Std. Error"], ncol = 1) #extract standard errors

summary(rep,"report") #report back what specified ADreport for
print(rep)
print(c("q=",q))
#print(best2)
print(best1)
#print(CoTS)





#===============================================================================
# Just for outputs
library(writexl)

modelOutputs <- cbind(years=years,
                      CoTS_age2p=CoTS[,3],
                      fastCoral=100*((Coral_f/(Km+Kf))),
                      slowCoral=100*((Coral_m/(Km+Kf)))
)

write_xlsx(as.data.frame(modelOutputs), "modelOutputs.xlsx")
