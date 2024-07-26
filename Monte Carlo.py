#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles = 1000  # Number of particles
box_size = 10.0       # Size of the box
num_steps = 10000     # Number of simulation steps
max_displacement = 0.1  # Maximum displacement per step

# Initialize particle positions
positions = np.random.uniform(0, box_size, (num_particles, 2))

# Run the simulation
for step in range(num_steps):
    particle_index = np.random.randint(0, num_particles)
    displacement = np.random.uniform(-max_displacement, max_displacement, 2)
    new_position = positions[particle_index] + displacement
    new_position = np.mod(new_position, box_size)
    positions[particle_index] = new_position

# Plot the results
plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], s=1)
plt.title('Monte Carlo Simulation of Particle Distribution in a Box')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, box_size)
plt.ylim(0, box_size)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# In[9]:


pip install pyrosetta-installer 


# In[13]:


pip install MDAnalysisTests


# In[1]:


import MDAnalysisTests


# In[2]:


import MDAnalysis as mda
from MDAnalysis.analysis import rms

# Step 1: Load trajectory and topology
# Replace with your actual trajectory and topology files
traj_file = '5GPX.xtc'
topology_file = '3cx9.pdb'

# Create a Universe object to represent the system (trajectory + topology)
u = mda.Universe(topology_file, traj_file)

# Step 2: Calculate RMSD
# Reference structure (typically the initial structure or a representative frame)
ref = u.select_atoms('protein')  # Select all protein atoms as reference
rmsd_analysis = rms.RMSD(u, ref, select='protein', align=True)

# Run the analysis
rmsd_analysis.run()

# Get RMSD values
rmsd_data = rmsd_analysis.rmsd

# Step 3: Visualize RMSD (optional)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(rmsd_data.time, rmsd_data.rmsd)
plt.xlabel('Time (ps)')
plt.ylabel('RMSD (Å)')
plt.title('Protein RMSD over Time')
plt.grid(True)
plt.show()


# In[4]:


pip install openmm


# In[6]:


pip install biopython


# In[7]:


pip install networkx


# In[1]:


from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder, Polypeptide

# Step 1: Download a PDB file (e.g., 3cx9)
pdb_file = "3cx9.pdb"

# Step 2: Retrieve sequence from PDB
parser = PDBParser(QUIET=True)
structure = parser.get_structure('3CX9', pdb_file)

# Extract sequence from the first chain
model = structure[0]
chain = model['A']  # Assuming chain A
sequence = Polypeptide.Polypeptide(chain).get_sequence()

print(f"Protein sequence (chain A): {sequence}")

# Step 3: Calculate molecular weight
mol_weight = Polypeptide.Polypeptide(sequence).molecular_weight()
print(f"Molecular weight: {mol_weight:.2f} g/mol")

# Step 4: Predict secondary structure
ppb = PPBuilder()
for pp in ppb.build_peptides(chain):
    print(f"Secondary structure for fragment {pp}:")
    for i, aa in enumerate(pp):
        print(f"Residue {i+1}: {aa}")
    print(f"Predicted secondary structure: {pp.get_ss()}\n")


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(num_samples):
    # Step 1: Generate random points
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    
    # Step 2: Count points inside the quarter circle
    inside_circle = x**2 + y**2 <= 1
    num_inside_circle = np.sum(inside_circle)
    
    # Step 3: Estimate pi
    pi_estimate = (num_inside_circle / num_samples) * 4
    return pi_estimate

# Run the simulation with 100,000 samples
num_samples = 100000
pi_estimate = monte_carlo_pi(num_samples)
print(f"Estimated value of pi: {pi_estimate}")

# Visualization (optional)
def plot_simulation(num_samples):
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    
    inside_circle = x**2 + y**2 <= 1
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x[inside_circle], y[inside_circle], color='blue', s=1, label='Inside Circle')
    plt.scatter(x[~inside_circle], y[~inside_circle], color='red', s=1, label='Outside Circle')
    plt.plot([0, 1], [0, 0], color='black')
    plt.plot([0, 0], [0, 1], color='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Monte Carlo Simulation with {num_samples} samples")
    plt.legend()
    plt.show()

# Plot the simulation with 10,000 samples
plot_simulation(10000)


# Problem Description
# We can estimate 
# 𝜋
# π by using the Monte Carlo method to simulate random points in a square and counting how many fall within a quarter circle inscribed within that square. The ratio of the number of points inside the quarter circle to the total number of points can be used to estimate 
# 𝜋
# π.
# 
# Steps to Solve the Problem
# Define the square and the quarter circle:
# 
# The square has a side length of 2 (from 
# (
# −
# 1
# ,
# −
# 1
# )
# (−1,−1) to 
# (
# 1
# ,
# 1
# )
# (1,1)).
# The quarter circle has a radius of 1, centered at the origin.
# Generate random points:
# 
# Randomly generate points within the square.
# Count points inside the quarter circle:
# 
# Use the equation of the circle 
# 𝑥
# 2
# +
# 𝑦
# 2
# ≤
# 1
# x 
# 2
#  +y 
# 2
#  ≤1 to check if a point is inside the quarter circle.
# Estimate 
# 𝜋
# π:
# 
# The ratio of points inside the quarter circle to the total number of points, multiplied by 4, gives an estimate of 
# 𝜋
# π.

# Explanation of the Code
# Generating Random Points:
# 
# We generate num_samples random points uniformly distributed in the square from 
# (
# −
# 1
# ,
# −
# 1
# )
# (−1,−1) to 
# (
# 1
# ,
# 1
# )
# (1,1).
# Checking Points Inside the Circle:
# 
# For each point 
# (
# 𝑥
# ,
# 𝑦
# )
# (x,y), we check if it lies inside the quarter circle using the condition 
# 𝑥
# 2
# +
# 𝑦
# 2
# ≤
# 1
# x 
# 2
#  +y 
# 2
#  ≤1.
# Estimating 
# 𝜋
# π:
# 
# The ratio of the number of points inside the quarter circle to the total number of points, multiplied by 4, provides an estimate of 
# 𝜋
# π.
# Visualization:
# 
# An optional function plot_simulation is provided to visualize the points and see how they distribute in relation to the quarter circle.
# This Monte Carlo simulation provides a simple yet powerful way to estimate the value of 
# 𝜋
# π. The accuracy of the estimate increases with the number of random samples generated.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.38e-23  # Boltzmann constant (J/K)
T = 300  # Temperature in Kelvin
m = 4.65e-26  # Mass of a gas molecule (kg), e.g., nitrogen molecule

# Step 1: Define the Maxwell-Boltzmann distribution
def maxwell_boltzmann_speed(v, T, m):
    return (m / (2 * np.pi * k_B * T))**(3/2) * 4 * np.pi * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

# Step 2: Generate random speeds using the Maxwell-Boltzmann distribution
def generate_speeds(num_samples, T, m):
    speeds = np.random.normal(0, np.sqrt(k_B * T / m), num_samples)
    return np.abs(speeds)

# Step 3: Compute kinetic energy
def kinetic_energy(speeds, m):
    return 0.5 * m * speeds**2

# Number of samples
num_samples = 100000

# Generate speeds
speeds = generate_speeds(num_samples, T, m)

# Compute kinetic energies
kinetic_energies = kinetic_energy(speeds, m)

# Estimate average kinetic energy
average_kinetic_energy = np.mean(kinetic_energies)
print(f"Estimated average kinetic energy: {average_kinetic_energy:.2e} J")

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(kinetic_energies, bins=100, density=True, alpha=0.6, color='g')
plt.title('Distribution of Kinetic Energies')
plt.xlabel('Kinetic Energy (J)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()


# Problem Description
# The kinetic energy of gas molecules follows the Maxwell-Boltzmann distribution. We can use Monte Carlo simulation to estimate the average kinetic energy of gas molecules at a given temperature.
# 
# Steps to Solve the Problem
# Define the Maxwell-Boltzmann distribution: This distribution describes the probability of molecules having a certain speed at a given temperature.
# Generate random speeds: Use the Maxwell-Boltzmann distribution to generate random speeds of the molecules.
# Compute kinetic energy: Calculate the kinetic energy of each molecule using the relation 
# 𝐸
# =
# 1
# 2
# 𝑚
# 𝑣
# 2
# E= 
# 2
# 1
# ​
#  mv 
# 2
#  .
# Estimate average kinetic energy: Compute the average kinetic energy of the molecules.

# Explanation of the Code
# Maxwell-Boltzmann Distribution: The function maxwell_boltzmann_speed defines the probability density function of speeds for gas molecules at temperature 
# 𝑇
# T and mass 
# 𝑚
# m.
# Generate Speeds: The function generate_speeds generates random speeds of molecules using a normal distribution and then takes the absolute value to ensure all speeds are positive.
# Compute Kinetic Energy: The function kinetic_energy computes the kinetic energy of each molecule using the formula 
# 𝐸
# =
# 1
# 2
# 𝑚
# 𝑣
# 2
# E= 
# 2
# 1
# ​
#  mv 
# 2
#  .
# Estimate Average Kinetic Energy: We compute the average kinetic energy of the molecules by taking the mean of the calculated kinetic energies.
# Visualization: We visualize the distribution of kinetic energies using a histogram.
# This Monte Carlo simulation provides an estimate of the average kinetic energy of gas molecules, which is an important property in understanding the behavior of gases and their thermodynamic properties.

# In[ ]:




