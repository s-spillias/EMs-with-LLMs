#include <TMB.hpp>
using namespace tmb;

// Model parameters
double SST; // Will be passed from data vectors

// Coral growth rates (per year)
double g_slow, g_fast;
// Coral carrying capacities (% cover)
double K_slow, K_fast;
// COTS feeding rates (individuals/m²/year)
double a_slow, a_fast;
// COTS mortality rates
double mu_COTS;
// Temperature sensitivity
double beta_SST;

// Initialize data vectors
vector<double> SST_dat = {"Data/timeseries_data_COTS_forcing.csv", "sst_dat"};
vector<double> cotsimm_dat = {"Data/timeseries_data_COTS_forcing.csv", "cotsimm_dat"};

vector<double> slow_dat = {"Data/timeseries_data_COTS_response.csv", "slow_dat"};
vector<double> fast_dat = {"Data/timeseries_data_COTS_response.csv", "fast_dat"};
vector<double> cots_dat = {"Data/timeseries_data_COTS_response.csv", "cots_dat"};

// Initialize state variables
T slow_pred, fast_pred, cots_pred;

// Model implementation
template<typename T>
T model(T data) {
    // Parameters
    T& params = *data.getParameters();
    
    // Initialize variables
    T sst = 25.0; // Base SST in °C
    T prev_slow = slow_dat[0] / 100.0; // Initial slow coral cover (as decimal)
    T prev_fast = fast_dat[0] / 100.0; // Initial fast coral cover (as decimal) 
    T prev_cots = cotsimm_dat[0];      // Initial COTS density
    
    // Main loop
    for(int t=0; t<data.nt(); t++) {
        // Store predictions from previous time step
        slow_pred = prev_slow;
        fast_pred = prev_fast;
        cots_pred = prev_cots;
        
        // Update SST from data
        if(t < data.n("SST_dat")) {
            sst += data["SST_dat"][t];
        }
        
        // Coral growth with resource limitation (using previous time step values)
        T slow_growth = g_slow * (1 - prev_slow / K_slow) * prev_slow;
        T fast_growth = g_fast * (1 - prev_fast / K_fast) * prev_fast;
        
        // COTS feeding using lagged coral cover
        T feeding = (a_slow * prev_slow + a_fast * prev_fast) / 
                    (1 + a_slow * prev_slow + a_fast * prev_fast);
        
        // Update COTS population with temperature effect and mortality
        T new_cots = prev_cots + feeding;
        new_cots *= (1 + beta_SST * (sst - 25));
        new_cots -= mu_COTS * prev_cots;
        
        // Apply Allee effect
        if(new_cots < 1e-8) new_cots = 1e-8; // Avoid negative values
        
        // Update state variables for next iteration
        prev_slow += slow_growth;
        prev_fast += fast_growth;
        prev_cots = new_cots;
        
        // Report predictions using lagged values
        REPORT("slow_pred", slow_pred * 100);  // Convert back to percentage
        REPORT("fast_pred", fast_pred * 100);
        REPORT("cots_pred", cots_pred);
    }
    
    return 0.0;
}
