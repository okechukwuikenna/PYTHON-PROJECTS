#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def calculate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return fingerprint

def calculate_similarity(smiles1, smiles2):
    fp1 = calculate_fingerprint(smiles1)
    fp2 = calculate_fingerprint(smiles2)
    return DataStructs.FingerprintSimilarity(fp1, fp2)

# Example usage
smiles1 = 'CCO'  # Ethanol
smiles2 = 'CCCO'  # Propanol

similarity = calculate_similarity(smiles1, smiles2)
print(f"Similarity between {smiles1} and {smiles2}: {similarity}")

