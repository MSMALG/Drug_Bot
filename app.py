import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# Load the processed data file
file_path = 'processed_bioactivity_data.csv'  # Update the path as needed
data = pd.read_csv(file_path)

# Streamlit app layout
st.title("Drug Molecule Recommendation Bot")
st.write("Enter the disease and target protein to get suggested drug molecules with side effect information.")

# Input for disease and target protein
disease_input = st.text_input("Enter Disease", "prostate cancer")
target_input = st.text_input("Enter Target Protein", "androgen receptor")

# Filter dataset based on disease and target
filtered_data = data[(data['disease'].str.lower() == disease_input.lower()) &
                     (data['target_name'].str.lower() == target_input.lower())]

# Display results if data is found for inputs
if not filtered_data.empty:
    st.write(f"### Suggested Drug Molecules for {disease_input.title()} with Target Protein {target_input.title()}")

    for idx, row in filtered_data.iterrows():
        st.write(f"**Molecule ID:** {row['molecule_chembl_id']}")
        
        # Show molecule structure image
        molecule = Chem.MolFromSmiles(row['canonical_smiles'])
        img = Draw.MolToImage(molecule, size=(300, 300))
        st.image(img, caption="Molecule Structure")

        # Display molecular properties
        st.write("**Properties:**")
        st.write(f"- Molecular Weight: {row['MoleculeWeight']}")
        st.write(f"- LogP: {row['LogP']}")
        st.write(f"- Number of H-Donors: {row['NumHDonors']}")
        st.write(f"- Number of H-Acceptors: {row['NumHAcceptors']}")
        
        # Check and display side effects
        if row['bioactivity'].lower() == 'active' and row['LogP'] > 5:
            st.warning("This molecule may have side effects such as increased heart rate.")
            
            # Suggesting a potential modification (for demo, reducing LogP)
            st.write("### Suggested Modification to Reduce Side Effects")
            modified_logp = max(row['LogP'] - 1, 0)  # Example adjustment
            st.write(f"- Adjusted LogP: {modified_logp}")

            # Display modified molecule image (for simplicity, using the same image in this demo)
            st.image(img, caption="Modified Molecule Structure")

        else:
            st.write("No significant side effects predicted.")

else:
    st.error("No matching data found. Please check the disease and target inputs.")
