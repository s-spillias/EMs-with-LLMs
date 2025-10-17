import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Define parameters
a = 0.2    # m^-1 day^-1
b = 0.2    # m^-1
c = 0.4    # m^2 g^-1 C^-1
e = 0.03   # g C m^-3
k = 0.05   # day^-1 (range: 0.0008-0.13)
q = 0.075  # day^-1 (range: 0.015-0.150)
r = 0.10   # day^-1 (range: 0.05-0.15)
s = 0.04   # day^-1 (range: 0.032-0.08)
N0 = 0.6   # g C m^-3 (range: 0.1-2.0)
alpha = 0.25  # (range: 0.2-0.5)
beta = 0.33
gamma = 0.5
lambda_ = 0.6  # day^-1
mu = 0.035  # g C m^-3

def npz_model(t, y):
    N, P, Z = y
    
    dN_dt = (- (N / (e + N)) * (a / (b + c * P)) * P 
             + r * P + (beta * lambda_ * P**2 / (mu**2 + P**2)) * Z 
             + gamma * q * Z + k * (N0 - N))
    
    dP_dt = ((N / (e + N)) * (a / (b + c * P)) * P 
             - r * P - (lambda_ * P**2 / (mu**2 + P**2)) * Z 
             - (s + k) * P)
    
    dZ_dt = ((alpha * lambda_ * P**2 / (mu**2 + P**2)) * Z - q * Z)
    
    return [dN_dt, dP_dt, dZ_dt]

# Initial conditions
N0_init = 0.4  # Initial N concentration
P0_init = 0.1  # Initial P concentration
Z0_init = 0.05 # Initial Z concentration
y0 = [N0_init, P0_init, Z0_init]

# Time span
tspan = (0, 100)  # Days
t_eval = np.linspace(tspan[0], tspan[1], 200)

# Solve the ODEs
sol = solve_ivp(npz_model, tspan, y0, t_eval=t_eval, method='RK45')

# Save results to CSV
data = {
    "Time (days)": sol.t,
    "N_dat (Nutrient concentration in g C m^-3)": sol.y[0],
    "P_dat (Phytoplankton concentration in g C m^-3)": sol.y[1],
    "Z_dat (Zooplankton concentration in g C m^-3)": sol.y[2]
}
df = pd.DataFrame(data)
# Round all numerical columns to 3 decimal places
df['Time (days)'] = df['Time (days)'].round(3)
df['N_dat (Nutrient concentration in g C m^-3)'] = df['N_dat (Nutrient concentration in g C m^-3)'].round(3)
df['P_dat (Phytoplankton concentration in g C m^-3)'] = df['P_dat (Phytoplankton concentration in g C m^-3)'].round(3)
df['Z_dat (Zooplankton concentration in g C m^-3)'] = df['Z_dat (Zooplankton concentration in g C m^-3)'].round(3)
df.to_csv("Data/NPZ_example/npz_model_response.csv", index=False)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='N_dat (Nutrients)')
plt.plot(sol.t, sol.y[1], label='P_dat (Phytoplankton)')
plt.plot(sol.t, sol.y[2], label='Z_dat (Zooplankton)')
plt.xlabel('Time (days)')
plt.ylabel('Concentration (g C m^-3)')
plt.legend()
plt.title('NPZ Model Dynamics')
plt.grid()
plt.show()
