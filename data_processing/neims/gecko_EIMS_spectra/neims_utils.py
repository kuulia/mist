from sys import version
from typing import NoReturn, Optional, List
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import numpy as np
import pandas as pd
import os

def get_py_ver() -> NoReturn:
    return version.replace(' ', '').split('|')[0].replace('.', '_')

def SafeMolFromSmiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except: 
        print('Issue with generating mol from smiles')
    return mol

def df_smiles_to_mol(df: pd.DataFrame) -> np.ndarray:
    print(f'NUM MOLS (SMILES): {len(df)}')
    mols = df['SMILES'].apply(SafeMolFromSmiles)\
                       .values\
                       .flatten()
    return mols

def file_exists(fpath: str):
    if os.path.exists(fpath + '/annotated.sdf'):
        #print("The file exists.")
        return True
    else:
        #print("The file does not exist.")
        return False
    
def rdkit_3d_exists(lines: str):
    found_rdkit = False
    found_3d = False
    for line in lines:
        if 'rdkit' in line:
            found_rdkit = True
        if '3d' in line:
            found_3d = True
    return found_rdkit and found_3d

def spectrum_exists(lines: str):
    spec_num_peaks = 0
    found_spec = False
    for _, line in enumerate(lines):
        if found_spec and line != '$$$$\n' and line != '\n':
            spec_num_peaks += 1
        if 'predicted spectrum' in line:
            found_spec = True  
    return found_spec, spec_num_peaks


def enumerate_folders(folder_nums: np.ndarray, path_to_folders: str, *, padder: str = '000000'):
    exists_idx = []
    not_exists_idx = []
    folders = []
    for idx, num in enumerate(folder_nums):
        default = padder
        folder = f''
        num_str = str(num)[::-1]
        for i, char in enumerate(default):
            try:
                folder += num_str[i]
            except:
                folder += '0'
        folder = folder[::-1]
        folders.append(folder)

    for idx, folder in enumerate(folders):
        if file_exists(f'{path_to_folders}/{folder}'):
            exists_idx.append(idx)
        else:
            not_exists_idx.append(idx)
    print(len(exists_idx), 'exists')
    print(len(not_exists_idx), 'does not exist')
    return folders, exists_idx, not_exists_idx

def parse_neims_spec_to_df(smiles_df: pd.DataFrame, 
                           folders, 
                           path_to_folder, *, 
                           not_exists_idx: Optional[List[str]] = None):
    if not_exists_idx is None:
        not_exists_idx = []
    df = pd.DataFrame()
    df['SMILES'] = smiles_df
    specs = []
    for idx, folder in enumerate(folders):
        if idx in not_exists_idx:
            specs.append(None)
            continue
        filename = f'{path_to_folder}/{folder}/annotated.sdf'
        with open(filename, 'r') as file:
            lines = file.readlines()
            lines = [line.lower() for line in lines]
        spectrum_found, spec_num_peaks = spectrum_exists(lines)
        assert rdkit_3d_exists(lines), 'rdkit 3d must exist'
        assert spectrum_found, 'spectrum must exist'
        assert spec_num_peaks >= 5, 'spectrum must have more at least 5 peaks'
        spec_flag = False
        spec = [[],[]]
        for line in lines:
            if 'predicted spectrum' in line:
                spec_flag = True
                continue
            if spec_flag and line != '$$$$\n' and line != '\n':
                location, intensity = line.replace('\n', '').split()
                spec[0].append(location); spec[1].append(intensity) 
        specs.append(np.array(spec, dtype=np.uint16).T)
    df['spec'] = specs
    return df

def smi_to_non_isomeric_smi(smi: str):
    mol = Chem.MolFromSmiles(smi)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
    return smi

def smi_to_non_isomeric_mol(smi: str):
    mol = Chem.MolFromSmiles(smi)
    smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
    mol = Chem.MolFromSmiles(smi)   
    return mol


def remove_duplicates_inchi(df_in: pd.DataFrame):
    df = df_in.copy()
    df = df.drop_duplicates(subset='SMILES')
    df['SMILES'] = df['SMILES'].apply(smi_to_non_isomeric_smi)
    df['mol'] = df['SMILES'].apply(smi_to_non_isomeric_mol)
    df['inchi'] = df['mol'].apply(Chem.MolToInchi)

    df = df.drop_duplicates(subset='inchi').reset_index().drop(columns='index')

    df = df.reset_index().drop(columns='index')

    return df

def mol_to_fp(mol, radius=2, nBits=2048):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)


def fp_to_tuple(fp):
    if fp is None:
        return None
    arr = np.zeros((1,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return tuple(arr)

def remove_duplicates_morgan(df_in: pd.DataFrame):
    df = df_in.copy()
    df['fingerprint'] = df['mol'].apply(mol_to_fp)
    df['fp_tuple'] = df['fingerprint'].apply(lambda fp: tuple(fp) if fp is not None else None)
    df = df.drop_duplicates(subset='fp_tuple').reset_index(drop=True).drop(columns=['fingerprint', 'fp_tuple'])
    return df

def make_labels_df(df_in: pd.DataFrame, 
                   ds_name: str, 
                   ionization_adduct: str, 
                   instrument: str):
    labels_df = pd.DataFrame()
    labels_df['smiles'] = df_in['SMILES']
    labels_df['formula'] = df_in['mol'].apply(CalcMolFormula)
    labels_df['spec'] = df_in['compound']
    labels_df['inchikey'] = df_in['mol'].apply(Chem.MolToInchiKey)
    labels_df['ionization'] = ionization_adduct
    labels_df['dataset'] = ds_name
    labels_df['instrument'] = instrument
    labels_df = labels_df[['dataset', 'spec', 'ionization', 'formula', 'smiles', 'inchikey', 'instrument']]
    return labels_df