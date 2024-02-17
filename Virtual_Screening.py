import streamlit as st
import pandas as pd
import subprocess
import os
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

st.set_page_config(
    page_title="Virtual Screening")


## function for calculating RDKit descriptors
    
def descriptors(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles] 
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()
    
    Mol_descriptors =[]
    for mol in mols:

        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors,desc_names 
    
    
    

## Loading the RF Model

def the_model(input_data):
    load_model = joblib.load('rf_model.pkl')
    # Make prediction
    prediction = load_model.predict(input_data)
    prediction_probability=load_model.predict_proba(input_data)
    
    x=pd.DataFrame(prediction_probability,columns=["Pa","Pia"])
    st.header('Prediction Result')
    
    prediction_output = pd.Series(prediction, name='Activity')
    #proba_output=pd.Series(prediction_probability,name="prediction_proba")
    
    molecule_name = pd.Series(reading_data[1], name='Molecule CHEMBL id/Molecule Name ')
    
    Result= pd.concat([molecule_name, prediction_output,x], axis=1)
    
    result = []
    for x in Result["Activity"]:
        if x==1:
            result.append("Active")
        if x==0:
            result.append("Inactive")
    Result["Activity"]=result
    st.write(Result)
    prediction_csv = Result.to_csv(index=False)
    st.download_button(label="Download prediction result",data=prediction_csv,file_name="My_result.csv")
st.title('BACE1 Activity Prediction App')
st.info('The BACE1 Activity Prediction App can be used to predict whether a  molecule is active or inactive for BACE1 target protein .')

st.header("Compound Virtual Screening")


uplouded_file=st.file_uploader("Please upload your input file", type=['txt'])


if st.button('Predict'):
    reading_data = pd.read_table(uplouded_file, sep='\t', header=None, names =["smiles","ID"])
    st.subheader('The input data')
    st.write(reading_data)


    st.subheader('Calculated RDKit Descriptors')
    # Read descriptor feature used in training
    Mol_descriptors,desc_names =descriptors(reading_data["smiles"])

    #st.download_button(label="Download PubchemFingerprinter subset Descriptor",data=desc_subset,file_name="Descriptor_subset.csv")
   
    df = pd.DataFrame(Mol_descriptors,columns=desc_names)

    st.write(df)
    
    
    the_model(Mol_descriptors)
else:
    st.warning('Limit 250 compounds per file')
    
    
