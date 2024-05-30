#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Potentiodynamic Polirization
import numpy as np
import matplotlib.pyplot as plt

# Simulate potential range (in volts)
potential = np.linspace(-1.0, 1.0, 1000)  # from -1V to 1V

# Hypothetical parameters for the polarization curve
exchange_current_density = 2e-6  # A/cm^2, assumed for ethane
beta_anodic = 0.12  # V/decade, assumed for ethane
beta_cathodic = 0.12  # V/decade, assumed for ethane

# Reference potential for the reaction (assumed value)
E0 = 0.0  # V

# Simulate the current density (in A/cm^2) using the Butler-Volmer equation
current_density = exchange_current_density * (np.exp((potential - E0) / beta_anodic) - np.exp(-(potential - E0) / beta_cathodic))

# Adding noise to simulate experimental data
noise = np.random.normal(0, 0.1e-6, potential.shape)
current_density_noisy = current_density + noise

# Plot the polarization curve
plt.figure(figsize=(10, 6))
plt.plot(potential, current_density_noisy, label='Ethane Polarization Curve', color='blue')
plt.xlabel('Potential (V)')
plt.ylabel('Current Density (A/cm²)')
plt.title('Potentiodynamic Polarization of Ethane')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Logarithmic scale for current density
plt.show()


# In[7]:


# Working with Methane:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the polarization data (assuming CSV format with columns 'Potential' and 'Current_Density')
data = pd.read_csv('polarization_data_methane.csv')
potential = data['Potential'].values
current_density = data['Current_Density'].values

# Define the Tafel equation for fitting
def tafel_eq_anodic(eta, i0, beta):
    return i0 * np.exp(eta / beta)

def tafel_eq_cathodic(eta, i0, beta):
    return -i0 * np.exp(-eta / beta)

# Select data for fitting (adjust these indices based on your data)
anodic_range = (potential > 0.2) & (potential < 0.6)
cathodic_range = (potential > -0.6) & (potential < -0.2)

# Fit the anodic branch
popt_anodic, _ = curve_fit(tafel_eq_anodic, potential[anodic_range], current_density[anodic_range])
i0_anodic, beta_anodic = popt_anodic

# Fit the cathodic branch
popt_cathodic, _ = curve_fit(tafel_eq_cathodic, potential[cathodic_range], current_density[cathodic_range])
i0_cathodic, beta_cathodic = popt_cathodic

# Calculate the corrosion current density (icorr)
icorr = np.sqrt(i0_anodic * i0_cathodic)

# Corrosion rate calculation (assuming uniform corrosion and using Faraday's law)
equivalent_weight = 27.92  # g/equiv (for iron, example)
density = 7.87  # g/cm³ (for iron, example)
area = 1.0  # cm² (example)

# Corrosion rate in mm/year
corrosion_rate = (icorr * 0.129) * (equivalent_weight / density)

# Plotting the polarization curve and Tafel fits
plt.figure(figsize=(10, 6))
plt.plot(potential, current_density, label='Experimental Data', color='blue')
plt.plot(potential[anodic_range], tafel_eq_anodic(potential[anodic_range], *popt_anodic), label='Anodic Fit', color='red', linestyle='--')
plt.plot(potential[cathodic_range], tafel_eq_cathodic(potential[cathodic_range], *popt_cathodic), label='Cathodic Fit', color='green', linestyle='--')
plt.xlabel('Potential (V)')
plt.ylabel('Current_Density (A/cm²)')
plt.title('Potentiodynamic Polarization of Methane')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print calculated parameters
print(f"Anodic Tafel slope (beta_anodic): {beta_anodic:.4f} V/decade")
print(f"Cathodic Tafel slope (beta_cathodic): {beta_cathodic:.4f} V/decade")
print(f"Anodic exchange current density (i0_anodic): {i0_anodic:.4e} A/cm²")
print(f"Cathodic exchange current density (i0_cathodic): {i0_cathodic:.4e} A/cm²")
print(f"Corrosion current density (icorr): {icorr:.4e} A/cm²")
print(f"Corrosion rate: {corrosion_rate:.4f} mm/year")


# In[8]:


import hardpotato as hp
import numpy as np
import matplotlib.pyplot as plt
import softpotato as sp
from scipy.optimize import curve_fit

##### Setup
# Select the potentiostat model to use:
# emstatpico, chi1205b, chi760e
#model = 'chi760e'
model = 'chi1205b'
#model = 'emstatpico'

# Initialization:
hp.potentiostat.Setup(model, path, folder)


##### Experimental parameters:
Eini = -0.3     # V, initial potential
Ev1 = 0.5       # V, first vertex potential
Ev2 = -0.3      # V, second vertex potential
Efin = -0.3     # V, final potential
dE = 0.001      # V, potential increment
nSweeps = 2     # number of sweeps
sens = 1e-4     # A/V, current sensitivity
header = 'CV'   # header for data file

##### Experiment:
sr = np.array([0.02, 0.05, 0.1, 0.2])          # V/s, scan rate
nsr = sr.size
i = []
for x in range(nsr):
    # initialize experiment:
    fileName = 'polarization_data_methane.csv' + str(int(sr[x]*1000)) + 'mVs'# base file name for data file
    cv = hp.potentiostat.CV(Eini, Ev1,Ev2, Efin, sr[x], dE, nSweeps, sens, fileName, header)
    # Run experiment:
    cv.run()
    # load data to do the data analysis later
    data = hp.load_data.CV(fileName + '.txt', folder, model)
    i.append(data.i)
i = np.array(i)
i = i[:,:,0].T
E = data.E


##### Data analysis
# Estimation of D with Randles-Sevcik
n = 1       # number of electrons
A = 0.071   # cm2, geometrical area
C = 1e-6    # mol/cm3, bulk concentration

# Showcases how powerful softpotato can be for fitting:
def DiffCoef(sr, D):
    macro = sp.Macro(n, A, C, D)
    rs = macro.RandlesSevcik(sr)
    return rs
    
iPk_an = np.max(i, axis=0)
iPk_ca = np.min(i, axis=0)
iPk = np.array([iPk_an, iPk_ca]).T
popt, pcov = curve_fit(DiffCoef, sr, iPk_an)
D = popt[0]

# Estimation of E0 from all CVs:
EPk_an = E[np.argmax(i, axis=0)]
EPk_ca = E[np.argmin(i, axis=0)]
E0 = np.mean((EPk_an+EPk_ca)/2)

#### Simulation with softpotato
iSim = []
for x in range(nsr):
    wf = sp.technique.Sweep(Eini,Ev1, sr[x])
    sim = sp.simulate.E(wf, n, A, E0, 0, C, D, D)
    sim.run()
    iSim.append(sim.i)
iSim = np.array(iSim).T
print(iSim.shape)
ESim = sim.E

##### Printing results
print('\n\n----------Results----------')
print('D = {:.2f}x10^-6 cm2/s'.format(D*1e6))
print('E0 = {:.2f} mV'.format(E0*1e3))

##### Plotting
srsqrt = np.sqrt(sr)
sp.plotting.plot(E, i*1e6, ylab='$i$ / $\mu$A', fig=1, show=0)
sp.plotting.plot(srsqrt, iPk*1e6, mark='o-', xlab=r'$\nu^{1/2}$ / V$^{1/2}$ s$^{-1/2}$', 
                 ylab='$i$ / $\mu$A', fig=2, show=0)

plt.figure(3)
plt.plot(E, i*1e6)
plt.plot(wf.E, iSim*1e6, 'k--')
plt.title('Experiment (-) vs Simulation (--)')
sp.plotting.format(xlab='$E$ / V', ylab='$i$ / $\mu$A', legend=[0], show=1)


# In[9]:


pip install hardpotato


# In[10]:


pip install softpotato


# In[ ]:


Potential,CurrentDensity
-1.0,-8.20E-06
-0.95,-7.50E-06
-0.9,-6.80E-06
-0.85,-6.10E-06
-0.8,-5.50E-06
-0.75,-4.90E-06
-0.7,-4.40E-06
-0.65,-3.90E-06
-0.6,-3.50E-06
-0.55,-3.10E-06
-0.5,-2.80E-06
-0.45,-2.50E-06
-0.4,-2.30E-06
-0.35,-2.10E-06
-0.3,-1.90E-06
-0.25,-1.70E-06
-0.2,-1.50E-06
-0.15,-1.30E-06
-0.1,-1.10E-06
-0.05,-9.00E-07
0.0,7.50E-07
0.05,9.00E-07
0.1,1.10E-06
0.15,1.30E-06
0.2,1.50E-06
0.25,1.70E-06
0.3,1.90E-06
0.35,2.10E-06
0.4,2.30E-06
0.45,2.50E-06
0.5,2.80E-06
0.55,3.10E-06
0.6,3.50E-06
0.65,3.90E-06
0.7,4.40E-06
0.75,4.90E-06
0.8,5.50E-06
0.85,6.10E-06
0.9,6.80E-06
0.95,7.50E-06
1.0,8.20E-06


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the polarization data (assuming CSV format with columns 'Potential' and 'CurrentDensity')
data = pd.read_csv('polarization_data_ethanol.csv')
potential = data['Potential'].values
current_density = data['CurrentDensity'].values

# Define the Tafel equation for fitting
def tafel_eq_anodic(eta, i0, beta):
    return i0 * np.exp(eta / beta)

def tafel_eq_cathodic(eta, i0, beta):
    return -i0 * np.exp(-eta / beta)

# Select data for fitting (adjust these indices based on your data)
anodic_range = (potential > 0.1) & (potential < 0.6)
cathodic_range = (potential > -0.6) & (potential < -0.1)

# Fit the anodic branch
popt_anodic, _ = curve_fit(tafel_eq_anodic, potential[anodic_range], current_density[anodic_range])
i0_anodic, beta_anodic = popt_anodic

# Fit the cathodic branch
popt_cathodic, _ = curve_fit(tafel_eq_cathodic, potential[cathodic_range], current_density[cathodic_range])
i0_cathodic, beta_cathodic = popt_cathodic

# Calculate the corrosion current density (icorr)
icorr = np.sqrt(i0_anodic * i0_cathodic)

# Corrosion rate calculation (assuming uniform corrosion and using Faraday's law)
equivalent_weight = 27.92  # g/equiv (for iron, example)
density = 7.87  # g/cm³ (for iron, example)
area = 1.0  # cm² (example)

# Corrosion rate in mm/year
corrosion_rate = (icorr * 0.129) * (equivalent_weight / density)

# Plotting the polarization curve and Tafel fits
plt.figure(figsize=(10, 6))
plt.plot(potential, current_density, label='Experimental Data', color='blue')
plt.plot(potential[anodic_range], tafel_eq_anodic(potential[anodic_range], *popt_anodic), label='Anodic Fit', color='red', linestyle='--')
plt.plot(potential[cathodic_range], tafel_eq_cathodic(potential[cathodic_range], *popt_cathodic), label='Cathodic Fit', color='green', linestyle='--')
plt.xlabel('Potential (V)')
plt.ylabel('Current Density (A/cm²)')
plt.title('Potentiodynamic Polarization of Ethanol')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print calculated parameters
print(f"Anodic Tafel slope (beta_anodic): {beta_anodic:.4f} V/decade")
print(f"Cathodic Tafel slope (beta_cathodic): {beta_cathodic:.4f} V/decade")
print(f"Anodic exchange current density (i0_anodic): {i0_anodic:.4e} A/cm²")
print(f"Cathodic exchange current density (i0_cathodic): {i0_cathodic:.4e} A/cm²")
print(f"Corrosion current density (icorr): {icorr:.4e} A/cm²")
print(f"Corrosion rate: {corrosion_rate:.4f} mm/year")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the polarization data (assuming CSV format with columns 'Potential' and 'CurrentDensity')
data = pd.read_csv('polarization_data_ethanol.csv')
potential = data['Potential'].values
current_density = data['CurrentDensity'].values

# Define the Tafel equation for fitting
def tafel_eq_anodic(eta, i0, beta):
    return i0 * np.exp(eta / beta)

def tafel_eq_cathodic(eta, i0, beta):
    return -i0 * np.exp(-eta / beta)

# Select data for fitting based on visual inspection
anodic_range = (potential > 0.1) & (potential < 0.6)
cathodic_range = (potential > -0.6) & (potential < -0.1)

# Fit the anodic branch
popt_anodic, _ = curve_fit(tafel_eq_anodic, potential[anodic_range], current_density[anodic_range])
i0_anodic, beta_anodic = popt_anodic

# Fit the cathodic branch
popt_cathodic, _ = curve_fit(tafel_eq_cathodic, potential[cathodic_range], current_density[cathodic_range])
i0_cathodic, beta_cathodic = popt_cathodic

# Calculate the corrosion current density (icorr)
icorr = np.sqrt(i0_anodic * i0_cathodic)

# Corrosion rate calculation (assuming uniform corrosion and using Faraday's law)
equivalent_weight = 27.92  # g/equiv (for iron, example)
density = 7.87  # g/cm³ (for iron, example)
area = 1.0  # cm² (example)

# Corrosion rate in mm/year
corrosion_rate = (icorr * 0.129) * (equivalent_weight / density)

# Plotting the polarization curve and Tafel fits
plt.figure(figsize=(10, 6))
plt.plot(potential, current_density, label='Experimental Data', color='blue')
plt.plot(potential[anodic_range], tafel_eq_anodic(potential[anodic_range], *popt_anodic), label='Anodic Fit', color='red', linestyle='--')
plt.plot(potential[cathodic_range], tafel_eq_cathodic(potential[cathodic_range], *popt_cathodic), label='Cathodic Fit', color='green', linestyle='--')
plt.xlabel('Potential (V)')
plt.ylabel('Current Density (A/cm²)')
plt.title('Potentiodynamic Polarization of Ethanol')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print calculated parameters
print(f"Anodic Tafel slope (beta_anodic): {beta_anodic:.4f} V/decade")
print(f"Cathodic Tafel slope (beta_cathodic): {beta_cathodic:.4f} V/decade")
print(f"Anodic exchange current density (i0_anodic): {i0_anodic:.4e} A/cm²")
print(f"Cathodic exchange current density (i0_cathodic): {i0_cathodic:.4e} A/cm²")
print(f"Corrosion current density (icorr): {icorr:.4e} A/cm²")
print(f"Corrosion rate: {corrosion_rate:.4f} mm/year")


# In[ ]:


#In the context of potentiodynamic polarization analysis, the primary parameters that may need adjustment to achieve accurate and reliable results include the following:
### Adjusted Python Script for Range Selection and Initial Guesses
#Here's an adjusted script incorporating these parameters with hypothetical initial guesses for the Tafel slopes and exchange current densities:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the polarization data (assuming CSV format with columns 'Potential' and 'CurrentDensity')
data = pd.read_csv('polarization_data_ethanol.csv')
potential = data['Potential'].values
current_density = data['CurrentDensity'].values

# Define the Tafel equations for fitting
def tafel_eq_anodic(eta, i0, beta):
    return i0 * np.exp(eta / beta)

def tafel_eq_cathodic(eta, i0, beta):
    return -i0 * np.exp(-eta / beta)

# Select data for fitting based on visual inspection (adjust these ranges as needed)
anodic_range = (potential > 0.1) & (potential < 0.6)
cathodic_range = (potential > -0.6) & (potential < -0.1)

# Provide initial guess values for the fitting parameters (adjust these as needed)
initial_guess_anodic = [1e-6, 0.1]  # Initial guess for i0 (A/cm²) and beta (V/decade)
initial_guess_cathodic = [1e-6, 0.1]  # Initial guess for i0 (A/cm²) and beta (V/decade)

# Fit the anodic branch
popt_anodic, _ = curve_fit(tafel_eq_anodic, potential[anodic_range], current_density[anodic_range], p0=initial_guess_anodic)
i0_anodic, beta_anodic = popt_anodic

# Fit the cathodic branch
popt_cathodic, _ = curve_fit(tafel_eq_cathodic, potential[cathodic_range], current_density[cathodic_range], p0=initial_guess_cathodic)
i0_cathodic, beta_cathodic = popt_cathodic

# Calculate the corrosion current density (icorr)
icorr = np.sqrt(i0_anodic * i0_cathodic)

# Corrosion rate calculation (assuming uniform corrosion and using Faraday's law)
equivalent_weight = 27.92  # g/equiv (for iron, example)
density = 7.87  # g/cm³ (for iron, example)
area = 1.0  # cm² (example)

# Corrosion rate in mm/year
corrosion_rate = (icorr * 0.129) * (equivalent_weight / density)

# Plotting the polarization curve and Tafel fits
plt.figure(figsize=(10, 6))
plt.plot(potential, current_density, label='Experimental Data', color='blue')
plt.plot(potential[anodic_range], tafel_eq_anodic(potential[anodic_range], *popt_anodic), label='Anodic Fit', color='red', linestyle='--')
plt.plot(potential[cathodic_range], tafel_eq_cathodic(potential[cathodic_range], *popt_cathodic), label='Cathodic Fit', color='green', linestyle='--')
plt.xlabel('Potential (V)')
plt.ylabel('Current Density (A/cm²)')
plt.title('Potentiodynamic Polarization of Ethanol')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

# Print calculated parameters
print(f"Anodic Tafel slope (beta_anodic): {beta_anodic:.4f} V/decade")
print(f"Cathodic Tafel slope (beta_cathodic): {beta_cathodic:.4f} V/decade")
print(f"Anodic exchange current density (i0_anodic): {i0_anodic:.4e} A/cm²")
print(f"Cathodic exchange current density (i0_cathodic): {i0_cathodic:.4e} A/cm²")
print(f"Corrosion current density (icorr): {icorr:.4e} A/cm²")
print(f"Corrosion rate: {corrosion_rate:.4f} mm/year")
```


