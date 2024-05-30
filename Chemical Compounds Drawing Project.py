#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Create a new Python file named chemistry_analysis.py. This script will read chemical structures, compute molecular properties, and visualize the molecules
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt

# Function to compute molecular properties
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES string: {smiles}")
        return None
    
    properties = {
        "Molecular Weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "Number of Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
        "Number of Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
    }
    return properties

# Function to visualize the molecule
def visualize_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Draw.MolToImage(mol)
    else:
        print(f"Invalid SMILES string: {smiles}")
        return None

def main():
    print("Welcome to the Chemistry Analysis Tool!")
    smiles = input("Enter a SMILES string of the molecule: ")

    # Compute and display molecular properties
    properties = compute_properties(smiles)
    if properties:
        print("Molecular Properties:")
        for prop, value in properties.items():
            print(f"{prop}: {value}")

    # Visualize the molecule
    image = visualize_molecule(smiles)
    if image:
        plt.imshow(image)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()


