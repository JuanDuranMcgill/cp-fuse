#!module load StdEnv/2020 gcc/9.3.0 opencv python/3.8 scipy-stack hdf5 geos/3.10.2 arrow/7.0.0

#!source ~/HIPT_Embedding_Env/bin/activate

import sys
import os

#sys.path.append(os.path.abspath('/home/sorkwos/projects/rrg-senger-ab/multimodality/contrastive_learning/tab-transformer-pytorch'))
sys.path.append(os.path.abspath('../../tab-transformer-pytorch'))
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import glob
import xmltodict
from IPython.display import display, HTML
import time as t
import random
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
import uuid
from datetime import datetime
from pycox.evaluation import EvalSurv

import torch

torch.set_num_threads(4)

# Function to drop columns with all same values or all NaN values
def drop_constant_columns(df):
    # Drop columns with all NaN values
    df = df.dropna(axis=1, how='all')

    # Find columns where all values are the same (including columns with a single unique non-NaN value)
    nunique = df.apply(pd.Series.nunique, dropna=False)
    cols_to_drop = nunique[nunique <= 1].index

    # Drop these columns
    df = df.drop(cols_to_drop, axis=1)

    return df

# Display the DataFrame in a scrollable format
def display_scrollable_dataframe(df, max_rows=20):
    display(HTML(df.to_html(max_rows=max_rows, classes='table table-striped table-bordered table-hover')))

# Function to recursively extract tags and values
def extract_tags_and_values(elem, parent_tag="", tag_count=None):
    if tag_count is None:
        tag_count = {}

    data = {}
    for child in elem:
        # Get the base tag without any namespaces
        base_tag = child.tag.split('}')[-1]

        # If this tag has a sequence attribute, append it to the tag
        sequence = child.attrib.get('sequence')

        # Construct full tag name, including parent if necessary
        if parent_tag:
            full_tag = f"{parent_tag}.{base_tag}"
        else:
            full_tag = base_tag

        # Handle repeated tags: append index or sequence to make the tag unique
        if sequence:
            full_tag += f"_seq_{sequence}"
        elif full_tag in tag_count:
            tag_count[full_tag] += 1
            full_tag += f"_{tag_count[full_tag]}"
        else:
            tag_count[full_tag] = 1

        # If the child has text and it's not just whitespace, store the value
        if child.text and child.text.strip():
            data[full_tag] = child.text.strip()

        # Recursively call the function to process the child elements
        data.update(extract_tags_and_values(child, full_tag, tag_count))

    return data

# Function to process all XML files in subfolders and consolidate them into a DataFrame
def process_xml_files(root_folder):
    all_data = []

    # Recursively find all XML files in the root folder and subfolders
    xml_files = glob.glob(os.path.join(root_folder, '**/*.xml'), recursive=True)

    for xml_file in xml_files:
        # Load and parse each XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract the patient data using the extract_tags_and_values function
        patient_data = extract_tags_and_values(root)

        # Add the patient data to the list
        all_data.append(patient_data)

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_data)

    # Optionally: Drop columns that have all NaN values or are constant
    df_cleaned = drop_constant_columns(df)

    return df_cleaned

# Function to extract and organize follow-up columns dynamically
def extract_and_organize_followup(df, prefix="follow_up_seq"):
    # Extract columns that match the follow-up sequence prefix
    follow_up_columns = [col for col in df.columns if prefix in col]
    
    # Extract unique sequence numbers dynamically (e.g., _1, _2, etc.)
    seq_numbers = sorted(set([col.split(f"{prefix}_")[1].split('_')[0] for col in follow_up_columns if f"{prefix}_" in col]))
    
    # Organize columns based on the sequence numbers
    organized_columns = []
    for seq in seq_numbers:
        seq_columns = [col for col in follow_up_columns if f"{prefix}_{seq}" in col]
        organized_columns.extend(seq_columns)
    
    return organized_columns

# Function to extract and organize drug columns, accounting for drug_1 being just 'patient.drugs.drug'
def extract_and_organize_drugs(df, base_prefix="patient.drugs.drug"):
    # Extract all drug-related columns (both drug_1 and drug_x)
    drug_columns = [col for col in df.columns if base_prefix in col]
    
    # Special handling for 'drug_1' columns without a suffix
    drug_1_columns = [col for col in drug_columns if base_prefix + "_" not in col]
    
    # Extract unique drug sequence numbers dynamically, handling possible extra parts in the column names
    drug_seq_numbers = sorted(set([int(col.split(f"{base_prefix}_")[1].split('.')[0])
                                   for col in drug_columns if f"{base_prefix}_" in col]))
    
    # Organize columns based on the sequence numbers
    organized_columns = drug_1_columns  # Start with drug_1 columns
    for seq in drug_seq_numbers:
        seq_columns = [col for col in drug_columns if f"{base_prefix}_{seq}." in col]
        organized_columns.extend(seq_columns)
    
    return organized_columns

def map_icd_to_site(icd_code):
    """Maps ICD-10 code to the corresponding AJCC site."""
    
    if icd_code in ['C02.9', 'C04.9', 'C06.9', 'C06.0', 'C03.9', 'C00.9', 'C05.0', 'C03.1', 'C04.0', 'C06.2', 'C02.1', 'C05.9', 'C03.0', 'C02.2']:
        return 'Section 3'
    
    elif icd_code in ['C32.9','C32.1']:
        return 'Section 5'
    
    elif icd_code == 'C14.8':
        return 'Section 9'
    
    elif icd_code in ['C09.9', 'C01', 'C10.9', 'C10.3','C13.9']:
        return 'Section 4'
    
    elif icd_code == 'C41.1':
        return 'Section 27'
    
    else:
        return 'Unknown Site'
    
def map_section_3_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    """Maps AJCC 6th edition to 7th edition for Section 3: Lip and Oral Cavity."""
    # T4 lesions have been divided into T4a (moderately advanced local disease) 
    # and T4b (very advanced local disease), leading to the stratification of 
    # Stage IV into Stage IVA, IVB, and IVC.
    if t_stage_6th == 'T4':
        return 'T4a', n_stage_6th, m_stage_6th  # T4 has been split into T4a and T4b
    else:
        return t_stage_6th, n_stage_6th, m_stage_6th


def map_section_4_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th, tissue_or_organ):
    """Maps AJCC 6th edition to 7th edition for Section 4 using the provided mappings."""
    
    if tissue_or_organ in ['Oral Tongue', 'Oral Cavity', 'Floor of mouth', 'Tonsil', 'Base of tongue', 'Buccal Mucosa', 
                           'Alveolar Ridge', 'Hard Palate', 'Lip', 'Oropharynx', 'Hypopharynx', 'Larynx']:
        # The conditions here should match your exact categories
        if t_stage_6th == 'T4':
            return 'T4a', n_stage_6th, m_stage_6th  # Modify based on specific stratification
        else:
            return t_stage_6th, n_stage_6th, m_stage_6th
    else:
        return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_5_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    """Maps AJCC 6th edition to 7th edition for Section 5: Larynx."""
    # T4 lesions have been divided into T4a (moderately advanced local disease) 
    # and T4b (very advanced local disease), leading to the stratification of 
    # Stage IV into Stage IVA, IVB, and IVC.
    if t_stage_6th == 'T4':
        return 'T4a', n_stage_6th, m_stage_6th  # T4 has been split into T4a and T4b
    else:
        return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_9_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    """Maps AJCC 6th edition to 7th edition for Section 9: Mucosal Melanoma of the Head and Neck."""
    # No changes needed for Section 9 as it didn't exist in the 6th edition.
    return t_stage_6th, n_stage_6th, m_stage_6th

def map_section_27_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th):
    """Maps AJCC 6th edition to 7th edition for Section 27: Bone (Mandible)."""
    # Stage III is reserved for G3 and G4 grades in the 7th edition.
    return t_stage_6th, n_stage_6th, m_stage_6th


    
def map_ajcc_6th_to_7th(icd_code, t_stage_6th, n_stage_6th, m_stage_6th, tissue_or_organ, grade=None):
    """Maps AJCC 6th edition to 7th edition based on the section derived from the ICD-10 code."""
    section = map_icd_to_site(icd_code)
    
    if section == 'Section 3':
        return map_section_3_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    elif section == 'Section 4':
        return map_section_4_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th, tissue_or_organ)
    elif section == 'Section 5':
        return map_section_5_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    elif section == 'Section 9':
        return map_section_9_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    elif section == 'Section 27':
        return map_section_27_6th_to_7th(t_stage_6th, n_stage_6th, m_stage_6th)
    else:
        return 'Unknown Section', t_stage_6th, n_stage_6th, m_stage_6th

def map_section_3_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    """Maps AJCC 5th edition to 6th edition for Section 3: Lip and Oral Cavity."""
    # T4 lesions have been divided into T4a (resectable) and T4b (unresectable), 
    # leading to the division of Stage IV into Stage IVA, IVB, and IVC.
    if t_stage_5th == 'T4':
        return 'T4a', n_stage_5th, m_stage_5th  # Assuming resectable; adjust if needed
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_4_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th, tissue_or_organ):
    """Maps AJCC 5th edition to 6th edition for Section 4: Pharynx, considering the specific subregions."""

    # For oropharynx and hypopharynx subdivisions:
    # T4 lesions are divided into T4a and T4b in the 6th edition.

    if tissue_or_organ in ['Oral Tongue', 'Oral Cavity', 'Floor of mouth', 'Tonsil', 'Base of tongue',
                           'Buccal Mucosa', 'Alveolar Ridge', 'Hard Palate', 'Lip', 'Oropharynx', 'Hypopharynx', 'Larynx']:
        if t_stage_5th == 'T4':
            return 'T4a', n_stage_5th, m_stage_5th  # Assuming resectable
        else:
            return t_stage_5th, n_stage_5th, m_stage_5th
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th


def map_section_5_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    """Maps AJCC 5th edition to 6th edition for Section 5: Larynx."""
    # T4 lesions have been divided into T4a (resectable) and T4b (unresectable),
    # leading to the division of Stage IV into Stage IVA, IVB, and IVC.
    if t_stage_5th == 'T4':
        return 'T4a', n_stage_5th, m_stage_5th  # Assuming resectable; adjust if needed
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_9_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    """Maps AJCC 5th edition to 6th edition for Section 9: Mucosal Melanoma of the Head and Neck."""
    # No changes needed for Section 9 as it did not exist in the 6th edition.
    return t_stage_5th, n_stage_5th, m_stage_5th

def map_section_27_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th):
    """Maps AJCC 5th edition to 6th edition for Section 27: Bone (Mandible)."""
    # M1 lesions have been divided into M1a and M1b:
    # - M1a is lung-only metastases (Stage IVA)
    # - M1b is metastases to other distant sites, including lymph nodes (Stage IVB).
    if m_stage_5th == 'M1':
        return t_stage_5th, n_stage_5th, 'M1a'  
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_ajcc_5th_to_6th(icd_code, t_stage_5th, n_stage_5th, m_stage_5th, tissue_or_organ):
    """Maps AJCC 5th edition to 6th edition based on the section derived from the ICD-10 code."""
    section = map_icd_to_site(icd_code)
    
    if section == 'Section 3':
        return map_section_3_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    elif section == 'Section 4':
        return map_section_4_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th, tissue_or_organ)
    elif section == 'Section 5':
        return map_section_5_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    elif section == 'Section 9':
        return map_section_9_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    elif section == 'Section 27':
        return map_section_27_5th_to_6th(t_stage_5th, n_stage_5th, m_stage_5th)
    else:
        return t_stage_5th, n_stage_5th, m_stage_5th

def map_clinical_5th_to_6th(row):
    # Clinical mapping
    clinical_t, clinical_n, clinical_m = map_ajcc_5th_to_6th(
        icd_code=row['patient.icd_10'],
        t_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    
    return pd.Series({
        'ajcc_clinical_t': clinical_t,
        'ajcc_clinical_n': clinical_n,
        'ajcc_clinical_m': clinical_m

    })

def map_clinical_6th_to_7th(row):
    # Clinical mapping
    clinical_t, clinical_n, clinical_m = map_ajcc_6th_to_7th(
        icd_code=row['patient.icd_10'],
        t_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    
    return pd.Series({
        'ajcc_clinical_t': clinical_t,
        'ajcc_clinical_n': clinical_n,
        'ajcc_clinical_m': clinical_m
    })

def map_clinical_and_pathologic_5th_to_6th(row):
    # Clinical mapping
    clinical_t, clinical_n, clinical_m = map_ajcc_5th_to_6th(
        icd_code=row['patient.icd_10'],
        t_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_5th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    
    # Pathologic mapping
    pathologic_t, pathologic_n, pathologic_m = map_ajcc_5th_to_6th(
        icd_code=row['patient.icd_10'],
        t_stage_5th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_T'],
        n_stage_5th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_N'],
        m_stage_5th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    
    return pd.Series({
        'patient.stage_event.tnm_categories.clinical_categories.clinical_T': clinical_t,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_N': clinical_n,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_M': clinical_m,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T': pathologic_t,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N': pathologic_n,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M': pathologic_m
    })

def map_clinical_and_pathologic_6th_to_7th(row):
    # Clinical mapping
    clinical_t, clinical_n, clinical_m = map_ajcc_6th_to_7th(
        icd_code=row['patient.icd_10'],
        t_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_T'],
        n_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_N'],
        m_stage_6th=row['patient.stage_event.tnm_categories.clinical_categories.clinical_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    
    # Pathologic mapping
    pathologic_t, pathologic_n, pathologic_m = map_ajcc_6th_to_7th(
        icd_code=row['patient.icd_10'],
        t_stage_6th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_T'],
        n_stage_6th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_N'],
        m_stage_6th=row['patient.stage_event.tnm_categories.pathologic_categories.pathologic_M'],
        tissue_or_organ=row['patient.anatomic_neoplasm_subdivision']
    )
    
    return pd.Series({
        'patient.stage_event.tnm_categories.clinical_categories.clinical_T': clinical_t,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_N': clinical_n,
        'patient.stage_event.tnm_categories.clinical_categories.clinical_M': clinical_m,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T': pathologic_t,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N': pathologic_n,
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M': pathologic_m
    })

# Example usage
#root_folder = './tcga_hnsc_xml_clinical'  # Specify your root folder here
#root_folder = '/Data/Juan/local_embeddings/TCGA-BRCA/clinical'
#root_folder = '/home/sorkwos/projects/rrg-senger-ab/multimodality/contrastive_learning/TCGA-HNSC-data/local_embeddings/TCGA-BRCA/clinical'
root_folder = '../local_embeddings/TCGA-BRCA/clinical'
# Process all XML files and get the consolidated DataFrame
df = process_xml_files(root_folder)

radiation_columns = extract_and_organize_drugs(df,"patient.radiations.radiation")
reordered_columns_df_1 = pd.concat([df.drop(columns=radiation_columns), df[radiation_columns]], axis=1)

drug_columns = extract_and_organize_drugs(reordered_columns_df_1)
reordered_columns_df_2 = pd.concat([reordered_columns_df_1.drop(columns=drug_columns), reordered_columns_df_1[drug_columns]], axis=1)

follow_up_columns = extract_and_organize_followup(reordered_columns_df_2)
reordered_columns_df_3 = pd.concat([reordered_columns_df_2.drop(columns=follow_up_columns), reordered_columns_df_2[follow_up_columns]], axis=1)

new_df = reordered_columns_df_3.loc[:, ~reordered_columns_df_3.columns.str.contains("patient.follow_ups|patient.radiations|patient.drugs")]

print("Total number of columns is",len(new_df.columns))
for c in new_df.columns:
    print(c)


new_df = new_df.drop(["admin.file_uuid", "admin.batch_number", "patient.patient_id"], axis=1)
#remove: admin.file_uuid, admin.batch_numer, patient.patient_id, 

new_df = new_df.drop(
    columns=[col for col in new_df.columns if 'new_tumor' in col] + [col for col in new_df.columns if 'additional_study' in col] + [col for col in new_df.columns if 'form_completion' in col] + [
        "patient.radiation_therapy",
        "patient.postoperative_rx_tx",
        "patient.vital_status",
        "patient.days_to_last_followup",
        "patient.days_to_death",
        "patient.history_of_neoadjuvant_treatment",
        "patient.person_neoplasm_cancer_status",
        "patient.age_at_initial_pathologic_diagnosis",
        "patient.year_of_initial_pathologic_diagnosis",
        "patient.tissue_source_site"
    ],
    axis=1
)

print("Total number of columns is",len(new_df.columns))
for c in new_df.columns:
    print(c)

#display_scrollable_dataframe(new_df)

new_df = new_df.dropna(axis=1, how='all')

print("Total number of columns is",len(new_df.columns))

clinical_data_updated = new_df.copy()


# Convert the string values to float, handling 'NaN'
clinical_data_updated['patient.days_to_birth'] = pd.to_numeric(clinical_data_updated['patient.days_to_birth'], errors='coerce')

clinical_data_updated['patient.days_to_birth'] = clinical_data_updated['patient.days_to_birth'].apply(lambda x: abs(x) if pd.notnull(x) and x < 0 else x)

clinical_data_final = clinical_data_updated

#clinical_data_final = clinical_data_updated.drop('patient.stage_event.system_version', axis=1)


for column in clinical_data_final.columns:
    print(f"Column: {column}")
    value_counts = clinical_data_final[column].value_counts(normalize=True,dropna=False) * 100
    for value, percentage in value_counts.items():
        print(f"  Value: {value}, Percentage: {percentage:.2f}%")
    print("\n" + "="*50 + "\n")

import pandas as pd

# Load the Excel file with endpoints
#file_path = "/Data/Juan/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
#file_path = "/home/sorkwos/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
file_path = "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81/TCGA-CDR-SupplementalTableS1.xlsx"
df = pd.read_excel(file_path)

df_blca = df[df['type'].str.contains("BRCA", na=False)]

# Load the Excel file with endpoints
#file_path = "/home/sorkwos/HIV_TCGA.xls"
#file_path = "/Data/Juan/HIV_TCGA.xls"
file_path = "HIV_TCGA.xls"
df_hiv_in= pd.read_excel(file_path, header=1)
df_hiv = df_hiv_in[df_hiv_in['Study'].str.contains("BRCA", na=False)]


#display_scrollable_dataframe(df_blca)

# Step 1: Select relevant columns from df_ucec
df_blca_filtered = df_blca[['bcr_patient_barcode', 'PFI', 'PFI.time']]

# Step 2: Merge the two DataFrames based on the patient barcode
clinical_data_final_updated = clinical_data_final.merge(
    df_blca_filtered,
    left_on='patient.bcr_patient_barcode',
    right_on='bcr_patient_barcode',
    how='left'  # Use 'left' to keep all rows from clinical_data_final and add matches from df_ucec
)

# Ensure SampleBarcode in df_hiv is sliced to first 12 characters
# Create a copy of df_hiv to ensure it's not a view of another DataFrame
df_hiv = df_hiv[['SampleBarcode', 'HPV load', 'HPV.status']].copy()

df_hiv['SampleBarcode_short'] = df_hiv['SampleBarcode'].str[:12]


# Perform the merge, keeping all rows from clinical_data_final_updated and filling non-matches with -1
clinical_data_final_updated = clinical_data_final_updated.merge(
    df_hiv[['SampleBarcode_short', 'HPV load', 'HPV.status']],
    how='left',
    left_on='patient.bcr_patient_barcode',
    right_on='SampleBarcode_short'
).drop(columns=['SampleBarcode_short'])

clinical_data_final_updated['HPV load'] = clinical_data_final_updated['HPV load'].fillna(-1)


# Step 3: Drop the redundant 'bar_patient_barcode' column after merging
clinical_data_final_updated = clinical_data_final_updated.drop(columns=['bcr_patient_barcode'])

# Display the updated DataFrame
#display_scrollable_dataframe(clinical_data_final_updated)
clinical_data_yes_id = clinical_data_final_updated.drop(columns=['patient.bcr_patient_uuid'])
clinical_data_yes = clinical_data_final_updated.drop(columns=['patient.bcr_patient_barcode','patient.bcr_patient_uuid'])

clinical_data_yes_id['patient.breast_carcinoma_immunohistochemistry_er_pos_finding_scale'] = \
    clinical_data_yes_id['patient.breast_carcinoma_immunohistochemistry_er_pos_finding_scale'].replace({
        '3 Point Scale': '3',
        '4 Point Scale': '4'
    })

clinical_data_yes_id['patient.breast_carcinoma_immunohistochemistry_progesterone_receptor_pos_finding_scale'] = \
    clinical_data_yes_id['patient.breast_carcinoma_immunohistochemistry_progesterone_receptor_pos_finding_scale'].replace({
        '3 Point Scale': '3',
        '4 Point Scale': '4'
    })


# Columns with "+" signs: remove the "+"
columns_with_plus = [
    'patient.her2_immunohistochemistry_level_result',
    'patient.immunohistochemistry_positive_cell_score',
    'patient.breast_carcinoma_immunohistochemistry_pos_cell_score'
]

for col in columns_with_plus:
    clinical_data_yes_id[col] = clinical_data_yes_id[col].str.replace('+', '', regex=False)

# Column: patient.her2_neu_breast_carcinoma_copy_analysis_input_total_number
# Replace "not amplified" with NaN and remove any signs like >, <, etc.
clinical_data_yes_id['patient.her2_neu_breast_carcinoma_copy_analysis_input_total_number'] = \
    clinical_data_yes_id['patient.her2_neu_breast_carcinoma_copy_analysis_input_total_number'].replace({
        'not amplified': np.nan
    })

# Remove any non-numeric characters except for the decimal point
clinical_data_yes_id['patient.her2_neu_breast_carcinoma_copy_analysis_input_total_number'] = \
    clinical_data_yes_id['patient.her2_neu_breast_carcinoma_copy_analysis_input_total_number'].replace({
        '>': '',
        '<': '',
        '=': ''
    }, regex=True)

# Drop extra columns from clinical_data_yes_id
clinical_data_yes_id = clinical_data_yes_id.drop(columns=["HPV load", "HPV.status"])

##############################
# FINAL SCRIPT: Standard CV 
##############################

# final_script.py

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import warnings
import itertools
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from pycox.models.loss import CoxPHLoss
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored


# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = device


###############################################################################
# Section A. Standard Functions for Survival Modeling (Clinical & WSI)
###############################################################################

class SurvivalDataset(Dataset):
    """
    Loads a JSON file of metadata and collects valid WSI embedding files
    (those with filenames containing the filter substring, e.g. "DX1").
    Each sample returns (regions, time, event), where regions is the tensor loaded from a .pt file.
    """
    def __init__(self, embedding_dir, json_path, filter_substring="DX1"):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        self.valid_samples = []
        self.filter_substring = filter_substring
        for entry in tqdm(self.samples, desc="Validating embeddings"):
            if self.filter_substring not in entry["file_name"]:
                continue
            pt_path = os.path.join(embedding_dir, entry["file_name"].replace(".svs", ".pt"))
            if os.path.exists(pt_path) and self.filter_substring in os.path.basename(pt_path):
                self.valid_samples.append({
                    "emb_path": pt_path,
                    "time_to_event": entry["time_to_event"],
                    "censoring": entry["censoring"]
                })
        print(f"Loaded {len(self.valid_samples)}/{len(self.samples)} valid slides with '{filter_substring}'")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        regions = torch.load(sample["emb_path"], map_location='cpu').float()
        return regions, sample["time_to_event"], sample["censoring"]

def collate_fn(batch):
    """
    Pads variable-length region embeddings in a batch to the maximum number of regions.
    Returns:
      - Padded embeddings: (batch_size, max_n, 192)
      - Boolean mask: (batch_size, max_n) (True for valid regions)
      - Times and events.
    """
    regions, times, events = zip(*batch)
    max_n = max(r.shape[0] for r in regions)
    padded_regions = []
    masks = []
    for r in regions:
        r = torch.tensor(r, dtype=torch.float32) if isinstance(r, np.ndarray) else r  # Convert NumPy array to Tensor
        pad_size = max_n - r.shape[0]
        padded = torch.cat([r, torch.zeros(pad_size, 192)], dim=0)
        padded_regions.append(padded)
        masks.append(torch.cat([torch.ones(r.shape[0]), torch.zeros(pad_size)]).bool())
    return (torch.stack(padded_regions).to(device),
            torch.stack(masks).to(device),
            torch.tensor(times).to(device),
            torch.tensor(events).to(device))

def contrastive_collate_fn(batch):
    """
    Custom collate function for contrastive learning.
    Pads variable-length region embeddings and creates a mask.
    
    Returns:
      - Padded embeddings for View 1: (batch_size, max_n, 192)
      - Padded embeddings for View 2: (batch_size, max_n, 192)
      - Boolean mask: (batch_size, max_n) (True for valid regions)
      - Labels: (batch_size,)
    """
    # Unpack correctly: batch = [((view1, view2), label), ...]
    embeddings, labels = zip(*batch)  # embeddings is a tuple of (view1, view2)

    # Separate view1 and view2
    view1_list, view2_list = zip(*embeddings)  # Now we have two lists of tensors

    # Find max number of regions in this batch
    max_n = max(v.shape[0] for v in view1_list)

    # Pad embeddings and create masks
    padded_view1, padded_view2, masks = [], [], []
    
    for v1, v2 in zip(view1_list, view2_list):
        pad_size = max_n - v1.shape[0]
        
        # Pad both views to max_n
        padded_v1 = torch.cat([v1, torch.zeros(pad_size, v1.shape[1])], dim=0)
        padded_v2 = torch.cat([v2, torch.zeros(pad_size, v2.shape[1])], dim=0)
        
        padded_view1.append(padded_v1)
        padded_view2.append(padded_v2)
        
        # Create mask (1 for real data, 0 for padded)
        masks.append(torch.cat([torch.ones(v1.shape[0]), torch.zeros(pad_size)]).bool())

    return (torch.stack(padded_view1).to(device),
            torch.stack(padded_view2).to(device),
            torch.stack(masks).to(device),
            torch.tensor(labels).to(device))

class AttentionAggregator(nn.Module):
    """
    Aggregates region-level embeddings using an attention mechanism.
    Input: (batch_size, max_n, 192)
    Output: (batch_size, 192)
    """
    def __init__(self, input_dim=192, hidden_dim=128):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, mask):
        
        #attn_logits = self.attention(x).squeeze(-1)  # (batch_size, max_n)
        hidden1 = self.attention[0](x)  # First Linear Layer Output
        hidden2 = torch.tanh(hidden1)  # After Tanh Activation
        attn_logits = self.attention[2](hidden2).squeeze(-1)  # Final Linear Layer
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        aggregated = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, 192)

        return aggregated

class SurvivalModel(nn.Module):
    """
    A survival model that aggregates region embeddings and outputs a risk score.
    """
    def __init__(self, input_dim=192, hidden_dim=128, dropout=0.3):
        super(SurvivalModel, self).__init__()
        self.aggregator = AttentionAggregator(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, 1)
    #def forward(self, x, mask):
        #x = self.aggregator(x, mask)
        #return self.fc(self.dropout(x)).squeeze(-1)

    def forward(self, x, mask):

        x = self.aggregator(x, mask)

        x = self.dropout(x)

        x = self.fc(x)



        return x.squeeze(-1)

# ------------------------------
# Define an augmentation function.
# This function receives an embedding (n, 192) and adds slight Gaussian noise.
# ------------------------------
def augment_fn(x):
    # x: (n, 192)
    noise = torch.randn_like(x) * 0.01
    return x + noise
'''
def train_evaluate_wsi(params, train_loader, val_loader):
    best_ibs_at_best_cindex = 0.0
    model = SurvivalModel(input_dim=192, hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = CoxPHLoss()
    best_cindex = 0
    epochs = params['epochs']

    for epoch in range(params['epochs']):
        total_loss = 0.0
        model.train()
        for x, mask, t, e in train_loader:
            optimizer.zero_grad()
            risk = model(x.to(device), mask.to(device))
            loss = loss_fn(risk, t.to(device), e.to(device))
            loss.backward()
            if params['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
            # --- Compute baseline cumulative hazard H_0(t) after each epoch --- #
            all_train_risks, all_train_times, all_train_events = [], [], []

            with torch.no_grad():
                for x, mask, t, e in train_loader:
                    risk = model(x.to(device), mask.to(device)).cpu().numpy()
                
                    all_train_risks.append(risk)
                    all_train_times.append(t.cpu().numpy())
                    all_train_events.append(e.cpu().numpy())

            all_train_risks = np.concatenate(all_train_risks)
            all_train_times = np.concatenate(all_train_times)
            all_train_events = np.concatenate(all_train_events)

            # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
            from lifelines import CoxPHFitter
            train_df = pd.DataFrame({
                'duration': all_train_times,
                'event': all_train_events,
                'risk': all_train_risks
            })
            cph = CoxPHFitter()
            cph.fit(train_df, duration_col='duration', event_col='event')
            baseline_hazard_df = cph.baseline_cumulative_hazard_

            def baseline_cumulative_hazard(t):
                if t in baseline_hazard_df.index:
                    return baseline_hazard_df.loc[t].values[0]
                else:
                    return baseline_hazard_df.values[-1, 0]

        model.eval()
        risks, times, events = [], [], []
        with torch.no_grad():
            for x, mask, t, e in val_loader:
                risks.append(model(x.to(device), mask.to(device)))
                times.append(t)
                events.append(e)


        # Concatenate tensors
        all_risks = torch.cat(risks)
        all_times = torch.cat(times)
        all_events = torch.cat(events)

        # Convert to numpy
        risks_val = all_risks.cpu().numpy()
        times_val = all_times.cpu().numpy()
        events_val = all_events.cpu().numpy().astype(bool)

        # --- Compute IPCW weights for censored patients --- #
        kmf = KaplanMeierFitter()
        kmf.fit(times_val, event_observed=events_val == 0)  # Only censored patients

        time_points = np.sort(np.unique(times_val))  # Use actual event times
        ipcw_weights = {}

        n = len(times_val)  # Sample size for truncation formula
        lower_bound = 5 / (np.sqrt(n) * np.log(n))  # Sample-size-based lower bound

        for t in time_points:
            G_t = kmf.survival_function_at_times(t).values[0]
            G_t = max(G_t, lower_bound)
            ipcw_weights[t] = 1 / np.clip(G_t, 1e-8, None)  # Avoid division by zero

        # --- Compute IBS directly --- #
        ibs_values = []

        for t in time_points:
            H_t = baseline_cumulative_hazard(t)  # Get H_0(t)
            #S_t = np.exp(-H_t * risks_val)  # Convert risk scores to survival probabilities
            S_t = np.exp(-H_t)  # DO NOT multiply again by risks_val
            #print(f"Epoch {epoch}: S_t min={S_t.min()}, max={S_t.max()}")
        

            # Apply IPCW only to censored patients
            censor_mask = (events_val == 0)  # Mask for censored patients
            brier_t = np.mean(
                (S_t - (times_val > t)) ** 2 * np.where(censor_mask, ipcw_weights[t], 1)
            )
            ibs_values.append(brier_t)

        # Compute final IBS as the mean of all Brier Scores across event times
        #print(ibs_values)
        ibs = np.mean(ibs_values)

        # --- Track Best C-index and corresponding IBS --- #

        risks = torch.cat(risks).cpu().numpy()
        times = torch.cat(times).cpu().numpy()
        events = torch.cat(events).cpu().numpy().astype(bool)
        #best_cindex = max(best_cindex, concordance_index(times, -risks, events))
        # Compute the C-index for this epoch
        cindex = concordance_index_censored(
            event_indicator=events,
            event_time=times,
            estimate=risks
        )[0]

        if cindex > best_cindex:
            best_cindex = cindex
            best_ibs_at_best_cindex = ibs


        # Print loss, validation C-index, and IBS every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Validation C-Index: {cindex:.4f} (Best: {best_cindex:.4f}) | IBS: {ibs:.4f} (IBS at Best C-Index: {best_ibs_at_best_cindex:.4f})")


    return best_cindex, best_ibs_at_best_cindex

'''

def train_evaluate_wsi(params, train_loader, val_loader, save_dir):
    best_ibs_at_best_cindex = 0.0
    model = SurvivalModel(input_dim=192, hidden_dim=params['hidden_dim'], dropout=params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = CoxPHLoss()
    best_cindex = 0
    epochs = params['epochs']

    for epoch in range(params['epochs']):
        total_loss = 0.0
        model.train()
        for x, mask, t, e in train_loader:


            if e.sum() == 0:
                print("Skipping batch with all censoredpythoin cases")
                continue  # Skip this batch
            optimizer.zero_grad()
            risk = model(x.to(device), mask.to(device))
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            loss = loss_fn(risk, t.to(device), e.to(device))
            loss.backward()
            if params['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        risks, times, events = [], [], []
        with torch.no_grad():
            for x, mask, t, e in val_loader:
                risks.append(model(x.to(device), mask.to(device)))
                times.append(t)
                events.append(e)


        # --- Track Best C-index and corresponding IBS --- #

        risks = torch.cat(risks).cpu().numpy()
        times = torch.cat(times).cpu().numpy()
        events = torch.cat(events).cpu().numpy().astype(bool)
        #best_cindex = max(best_cindex, concordance_index(times, -risks, events))
        # Compute the C-index for this epoch
        cindex = concordance_index_censored(
            event_indicator=events,
            event_time=times,
            estimate=risks
        )[0]

        if cindex>best_cindex:
            if 'best_model_path' in locals() and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_cindex=cindex
            su1=uuid.uuid4().hex[:6]
            su2=uuid.uuid4().hex[:6]
            best_model_path=os.path.join(save_dir,f"best_model_epoch_{epoch}_{su1}_{su2}.pth")
            torch.save(model.state_dict(),best_model_path)

        # Print loss, validation C-index, and IBS every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Validation C-Index: {cindex:.4f} (Best: {best_cindex:.4f})")


    # Compute IBS only for the best model
    model.load_state_dict(torch.load(best_model_path))


    # --- Compute baseline cumulative hazard H_0(t) after each epoch --- #
    all_train_risks, all_train_times, all_train_events = [], [], []

    with torch.no_grad():
        for x, mask, t, e in train_loader:

            risk = model(x.to(device), mask.to(device))
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            risk = risk.cpu().numpy()
        
            all_train_risks.append(risk)
            all_train_times.append(t.cpu().numpy())
            all_train_events.append(e.cpu().numpy())

    all_train_risks = np.concatenate(all_train_risks)
    all_train_times = np.concatenate(all_train_times)
    all_train_events = np.concatenate(all_train_events)

    # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
    from lifelines import CoxPHFitter
    train_df = pd.DataFrame({
        'duration': all_train_times,
        'event': all_train_events,
        'risk': all_train_risks
    })
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='duration', event_col='event')
    baseline_hazard_df = cph.baseline_cumulative_hazard_

    def baseline_cumulative_hazard(t):
        if t in baseline_hazard_df.index:
            return baseline_hazard_df.loc[t].values[0]
        else:
            return baseline_hazard_df.values[-1, 0]


    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for x, mask, t, e in val_loader:
            risks.append(model(x.to(device), mask.to(device)))
            times.append(t)
            events.append(e)


    # Concatenate tensors
    all_risks = torch.cat(risks)
    all_times = torch.cat(times)
    all_events = torch.cat(events)

    # Convert to numpy
    risks_val = all_risks.cpu().numpy()
    times_val = all_times.cpu().numpy()
    events_val = all_events.cpu().numpy().astype(bool)

     # Step 1: Prepare data
    risk_scores = risks_val  # Your risk scores
    times = times_val  # Event times
    events = events_val  # Event indicators (1 = event, 0 = censored)

    risk_scores = np.clip(risk_scores, -4, 4)  # First: control exp(risk)  
    # Step 2: Estimate baseline survival function
    baseline_surv = cph.baseline_survival_
    time_points = baseline_surv.index.values
    baseline_survival_at_times = baseline_surv['baseline survival'].values
    # Step 3: Compute time-dependent survival probabilities
    time_dependent_survival = np.array([
        baseline_survival_at_times ** np.exp(score) for score in risk_scores
    ])

    # Step 4: Create a DataFrame of survival probabilities
    # Each row corresponds to a patient, and each column corresponds to a time point
    survival_df = pd.DataFrame(time_dependent_survival, columns=time_points)

    #survival_df = survival_df.clip(lower=0.0, upper=1.0)
    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  
    survival_df = survival_df.apply(
        lambda row: pd.Series(np.minimum.accumulate(row.values), index=row.index),
        axis=1
    )

    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  

    # Step 5: Remove the last quartile of time points
    time_grid = time_points[time_points <= np.quantile(times[events == 1], 0.75)]

    survival_df.columns = survival_df.columns.astype(np.float32)
    time_grid = time_grid.astype(np.float32)

    # Step 6: Compute IBS using EvalSurv
    # Transpose survival_df to match EvalSurv's expected format
    ev = EvalSurv(survival_df.T, times, events, censor_surv='km')
    ibs = min(ev.integrated_brier_score(time_grid), 0.25)  # Compute IBS
    best_ibs = ibs
    print(f"\nBest Validation C-Index: {best_cindex:.4f}")
    print(f"Corresponding IBS: {ibs:.4f}")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)


    return best_cindex, ibs


class SurvivalModel_clinical(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim=64,
        depth=6,
        heads=8,
        attn_dropout=0.3,
        ff_dropout=0.3
    ):
        super().__init__()
        from tab_transformer_pytorch import FTTransformer
        self.ft_transformer = FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        self.survival_head = nn.Linear(dim, 1)

    def forward(self, x_cat, x_num):
        embeddings = self.ft_transformer(x_cat, x_num, return_embedding=True)
        return self.survival_head(embeddings).squeeze(-1)




class LateFusionSurvivalModel(nn.Module):
    """
    Late fusion model that:
    - Uses the clinical branch (FT-Transformer) to output a risk score.
    - Uses the WSI branch (attention aggregator) to output a risk score.
    - Fuses the two risk scores via a trainable weighted sum.
    """
    def __init__(self, clinical_model, wsi_model, fusion_module):
        super(LateFusionSurvivalModel, self).__init__()
        self.clinical_model = clinical_model
        self.wsi_model = wsi_model
        self.fusion_module = fusion_module  # This operates on risk scores

    def forward(self, x_cat, x_num, x_wsi, mask_wsi):
        """
        Forward pass:
        - Extract risk scores from clinical & WSI models separately.
        - Fuse them via trainable weight fusion.
        """
        risk_clin = self.clinical_model(x_cat, x_num)  # Clinical risk score (B,)
        risk_wsi = self.wsi_model(x_wsi, mask_wsi)     # WSI risk score (B,)
        risk_fused = self.fusion_module(risk_clin, risk_wsi)  # Trainable weighted fusion of risk scores
        return risk_fused


class TrainableWeightFusion(nn.Module):
    """
    Learns trainable scalar weights (α, β) to balance the contribution of clinical and WSI risk scores.
    Late fusion operates on the predicted risks, not the embeddings.
    """
    def __init__(self):
        super(TrainableWeightFusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Initialize α
        self.beta = nn.Parameter(torch.tensor(0.5))   # Initialize β
    
    def forward(self, risk_clin, risk_wsi):
        """
        risk_clin: (B,) - Clinical risk scores
        risk_wsi: (B,) - WSI risk scores
        """
        risk_fused = self.alpha * risk_clin + self.beta * risk_wsi  # Weighted sum of risk scores
        return risk_fused

class MetaLearnerFusion(nn.Module):
    """
    Late Fusion using a Meta-Learner (MLP) to learn a non-linear combination 
    of the clinical and WSI risk scores.
    """
    def __init__(self, hidden_dim=16):
        super(MetaLearnerFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Takes (risk_clin, risk_wsi) as input
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs fused risk score
        )

    def forward(self, risk_clin, risk_wsi):
        """
        risk_clin: (B,) - Clinical risk scores
        risk_wsi: (B,) - WSI risk scores
        """
        risk_input = torch.stack([risk_clin, risk_wsi], dim=1)  # Shape (B, 2)
        risk_fused = self.mlp(risk_input).squeeze(-1)  # Shape (B,)
        return risk_fused




#############################################
# Unified End-to-End Training Function
#############################################

'''
def train_evaluate_end_to_end_fusion_late(
    clinical_train,
    clinical_val,
    ds_orig_train,
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    device='cuda:1'
):
    """
    End-to-end training with Late Fusion (Risk Score Fusion).
    Uses trainable weights or a meta-learner to fuse the risk scores.
    """
    # --- Process Clinical Data ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val = scaler.transform(clinical_val[numeric_cols])
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat = clinical_val[categorical_cols]

    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat = torch.tensor(val_cat.values, dtype=torch.long)
    X_val_num = torch.tensor(val_num.values, dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train = clinical_train["PFI"].values.astype(bool)
    time_val = clinical_val["PFI.time"].values.astype(float)
    event_val = clinical_val["PFI"].values.astype(bool)

    # --- Build Unified Datasets ---
    unified_train_dataset = UnifiedDataset(
        clin_cat=X_train_cat,
        clin_num=X_train_num,
        times=time_train,
        events=event_train,
        wsi_dataset=ds_orig_train
    )
    unified_val_dataset = UnifiedDataset(
        clin_cat=X_val_cat,
        clin_num=X_val_num,
        times=time_val,
        events=event_val,
        wsi_dataset=ds_orig_val
    )
    batch_size = clin_param.get('batch_size', 64)
    unified_train_loader = DataLoader(unified_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=unified_collate_fn)
    unified_val_loader = DataLoader(unified_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)

    # --- Instantiate Branches & Fusion Modules ---
    cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]
    num_cont = len(numeric_cols)

    # Clinical branch parameters
    lr_clin = clin_param.get("lr", 1e-4)
    epochs_clin = clin_param.get("epochs", 20)
    dim_val = clin_param.get("dim", 64)
    depth_val = clin_param.get("depth", 6)
    heads_val = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val = clin_param.get("ff_dropout", 0.3)

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # WSI branch parameters
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_dropout = wsi_param.get('dropout', 0.0)
    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    # **NEW LATE FUSION MODULE** (Trainable Weights or Meta-Learner)
    fusion_module = MetaLearnerFusion().to(device)
    #fusion_module = TrainableWeightFusion().to(device)

    # Late Fusion Model (Fusing risk scores, NOT embeddings)
    late_fusion_model = LateFusionSurvivalModel(
        clinical_model=clinical_model,
        wsi_model=wsi_model,
        fusion_module=fusion_module  # Now working on risk scores
    ).to(device)

    # --- Define Optimizer & Loss ---
    optimizer = torch.optim.Adam(late_fusion_model.parameters(), lr=lr_clin)
    loss_fn = CoxPHLoss()

    # ----- End-to-End Training Loop -----
    unified_epochs = max(epochs_clin, wsi_param.get('epochs', 50), 80)
    best_cindex = 0
    best_ibs_at_best_cindex = 0

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(unified_epochs):
        late_fusion_model.train()
        total_loss = 0.0
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
            t_b, e_b = t_b.to(device), e_b.to(device)

            if e_b.sum() == 0:
                print("Skipping batch with all censored cases")
                continue  # Skip this batch

            optimizer.zero_grad()
            risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi)
            loss = loss_fn(risk, t_b, e_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            # --- Compute baseline cumulative hazard H_0(t) after each epoch --- #
            all_train_risks, all_train_times, all_train_events = [], [], []

            with torch.no_grad():
                for batch in unified_train_loader:
                    x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                    x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)


                    risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi).cpu().numpy()

                    all_train_risks.append(risk)
                    all_train_times.append(t_b.cpu().numpy())
                    all_train_events.append(e_b.cpu().numpy())

            all_train_risks = np.concatenate(all_train_risks)
            all_train_times = np.concatenate(all_train_times)
            all_train_events = np.concatenate(all_train_events)

            # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
            from lifelines import CoxPHFitter
            train_df = pd.DataFrame({
                'duration': all_train_times,
                'event': all_train_events,
                'risk': all_train_risks
            })
            cph = CoxPHFitter()
            cph.fit(train_df, duration_col='duration', event_col='event')
            baseline_hazard_df = cph.baseline_cumulative_hazard_

            def baseline_cumulative_hazard(t):
                if t in baseline_hazard_df.index:
                    return baseline_hazard_df.loc[t].values[0]
                else:
                    return baseline_hazard_df.values[-1, 0]



        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(unified_train_loader)
            # ----- Validation after each epoch -----
            late_fusion_model.eval()
            risks_list, times_list, events_list = [], [], []
            with torch.no_grad():
                for batch in unified_val_loader:
                    x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                    x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
                    risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi)
                    risks_list.append(risk.cpu())
                    times_list.append(t_b)
                    events_list.append(e_b)
            all_risks = torch.cat(risks_list)
            all_times = torch.cat(times_list)
            all_events = torch.cat(events_list)

            # Convert to numpy
            risks_val = all_risks.cpu().numpy()
            times_val = all_times.cpu().numpy()
            events_val = all_events.cpu().numpy().astype(bool)

            # --- Compute IPCW weights for censored patients --- #
            kmf = KaplanMeierFitter()
            kmf.fit(times_val, event_observed=events_val == 0)  # Only censored patients

            time_points = np.sort(np.unique(times_val))  # Use actual event times
            ipcw_weights = {}

            n = len(times_val)  # Sample size for truncation formula
            lower_bound = 5 / (np.sqrt(n) * np.log(n))  # Sample-size-based lower bound

            for t in time_points:
                G_t = kmf.survival_function_at_times(t).values[0]
                G_t = max(G_t, lower_bound)
                ipcw_weights[t] = 1 / np.clip(G_t, 1e-8, None)  # Avoid division by zero

            # --- Compute IBS directly --- #
            ibs_values = []

            for t in time_points:
                H_t = baseline_cumulative_hazard(t)  # Get H_0(t)
                #S_t = np.exp(-H_t * risks_val)  # Convert risk scores to survival probabilities
                S_t = np.exp(-H_t)  # DO NOT multiply again by risks_val
                #print(f"Epoch {epoch}: S_t min={S_t.min()}, max={S_t.max()}")
            

                # Apply IPCW only to censored patients
                censor_mask = (events_val == 0)  # Mask for censored patients
                brier_t = np.mean(
                    (S_t - (times_val > t)) ** 2 * np.where(censor_mask, ipcw_weights[t], 1)
                )
                ibs_values.append(brier_t)

            # Compute final IBS as the mean of all Brier Scores across event times
            #print(ibs_values)
            ibs = np.mean(ibs_values)


            cindex = concordance_index_censored(
                event_indicator=events_val,
                event_time=times_val,
                estimate=risks_val
            )[0]

            if cindex > best_cindex:
                best_cindex = cindex
                best_ibs_at_best_cindex = ibs

            print(f"[Late Fusion] Epoch {epoch+1}/{unified_epochs}, Loss: {avg_loss:.4f}, Current Val C-index: {cindex:.4f} | IBS: {ibs:.4f} (IBS at Best C-Index: {best_ibs_at_best_cindex:.4f})")

    print(f"[Late Fusion] Best Validation C-Index: {best_cindex:.4f}")
    return best_cindex, best_ibs_at_best_cindex




'''



def train_evaluate_end_to_end_fusion_late(
    cp_model,
    clinical_train,
    clinical_val,
    ds_orig_train,
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    save_dir,
    device='cuda:1'
):
    """
    End-to-end training with Late Fusion (Risk Score Fusion).
    Uses trainable weights or a meta-learner to fuse the risk scores.
    """
    # --- Process Clinical Data ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val = scaler.transform(clinical_val[numeric_cols])
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat = clinical_val[categorical_cols]

    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat = torch.tensor(val_cat.values, dtype=torch.long)
    X_val_num = torch.tensor(val_num.values, dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train = clinical_train["PFI"].values.astype(bool)
    time_val = clinical_val["PFI.time"].values.astype(float)
    event_val = clinical_val["PFI"].values.astype(bool)

    # --- Build Unified Datasets ---
    unified_train_dataset = UnifiedDataset(
        clin_cat=X_train_cat,
        clin_num=X_train_num,
        times=time_train,
        events=event_train,
        wsi_dataset=ds_orig_train
    )
    unified_val_dataset = UnifiedDataset(
        clin_cat=X_val_cat,
        clin_num=X_val_num,
        times=time_val,
        events=event_val,
        wsi_dataset=ds_orig_val
    )
    batch_size = clin_param.get('batch_size', 64)
    unified_train_loader = DataLoader(unified_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=unified_collate_fn)
    unified_val_loader = DataLoader(unified_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)

    # --- Instantiate Branches & Fusion Modules ---
    cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]
    num_cont = len(numeric_cols)

    # Clinical branch parameters
    lr_clin = clin_param.get("lr", 1e-4)
    epochs_clin = clin_param.get("epochs", 20)
    dim_val = clin_param.get("dim", 64)
    depth_val = clin_param.get("depth", 6)
    heads_val = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val = clin_param.get("ff_dropout", 0.3)

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # WSI branch parameters
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_dropout = wsi_param.get('dropout', 0.0)
    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    # **NEW LATE FUSION MODULE** (Trainable Weights or Meta-Learner)
    fusion_module = MetaLearnerFusion().to(device)
    if cp_model == "trainableweights":
        print("TRAINING TRAINABLE WEIGHTS")
        fusion_module = TrainableWeightFusion().to(device)
    elif cp_model == "metalearning":
        print("TRAINING METALEARNING")
        fusion_module = MetaLearnerFusion().to(device)


    # Late Fusion Model (Fusing risk scores, NOT embeddings)
    late_fusion_model = LateFusionSurvivalModel(
        clinical_model=clinical_model,
        wsi_model=wsi_model,
        fusion_module=fusion_module  # Now working on risk scores
    ).to(device)

    # --- Define Optimizer & Loss ---
    optimizer = torch.optim.Adam(late_fusion_model.parameters(), lr=lr_clin)
    loss_fn = CoxPHLoss()

    # ----- End-to-End Training Loop -----
    unified_epochs = max(epochs_clin, wsi_param.get('epochs', 50), 80)
    best_cindex = 0
    #best_ibs_at_best_cindex = 0

    # Enable anomaly detection
    #torch.autograd.set_detect_anomaly(True)
    best_model_path = os.path.join(save_dir, "best_late_fusion_model.pth")
    os.makedirs(save_dir, exist_ok=True)

    # Train
    for epoch in range(unified_epochs):
        late_fusion_model.train()
        total_loss = 0.0

        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            if e_b.sum() == 0:
                print("Skipping batch with all censoredpythoin cases")
                continue  # Skip this batch
            x_cat, x_num, x_wsi, mask_wsi = (
                x_cat.to(device),
                x_num.to(device),
                x_wsi.to(device),
                mask_wsi.to(device)
            )
            t_b, e_b = t_b.to(device), e_b.to(device)

            # If a batch has all censored cases, skip
            if e_b.sum() == 0:
                continue

            optimizer.zero_grad()
            risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi)
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            loss = loss_fn(risk, t_b, e_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- Validation (C-index) ----
        late_fusion_model.eval()
        risks_list, times_list, events_list = [], [], []
        with torch.no_grad():
            for batch in unified_val_loader:
                x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                x_cat, x_num, x_wsi, mask_wsi = (
                    x_cat.to(device),
                    x_num.to(device),
                    x_wsi.to(device),
                    mask_wsi.to(device)
                )
                risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi)
                risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
                risk = torch.clamp(risk, -1e6, 1e6)   
                risks_list.append(risk.cpu())
                times_list.append(t_b)
                events_list.append(e_b)

        # Concatenate validation results
        all_risks = torch.cat(risks_list).numpy()
        all_times = torch.cat(times_list).numpy()
        all_events = torch.cat(events_list).numpy().astype(bool)

        cindex = concordance_index_censored(
            event_indicator=all_events,
            event_time=all_times,
            estimate=all_risks
        )[0]

        # Check if this is the best model so far; if so, save weights
        if cindex>best_cindex:
            if 'best_model_path' in locals() and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_cindex=cindex
            su1=uuid.uuid4().hex[:6]
            su2=uuid.uuid4().hex[:6]
            best_model_path=os.path.join(save_dir,f"best_model_epoch_{epoch}_{su1}_{su2}.pth")
            torch.save(late_fusion_model.state_dict(),best_model_path)


        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(unified_train_loader)
            print(f"[Late Fusion] Epoch {epoch+1}/{unified_epochs}, Loss: {avg_loss:.4f}, "
                  f"Val C-Index: {cindex:.4f} (Best: {best_cindex:.4f})")

    print(f"\n[Late Fusion] Training complete. Best Validation C-Index: {best_cindex:.4f}\n")

    # ----------------------------------------------------------------------
    #        Compute IBS only for the best model (just like your first fn)
    # ----------------------------------------------------------------------
    # 1) Load the best model
    late_fusion_model.load_state_dict(torch.load(best_model_path))
    late_fusion_model.eval()

    # 2) Get training set risk scores to fit a baseline hazard (using lifelines CoxPHFitter)
    all_train_risks, all_train_times, all_train_events = [], [], []
    with torch.no_grad():
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = (
                x_cat.to(device),
                x_num.to(device),
                x_wsi.to(device),
                mask_wsi.to(device)
            )
            risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi)
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            all_train_risks.append(risk.cpu().numpy())
            all_train_times.append(t_b.numpy())
            all_train_events.append(e_b.numpy())

    all_train_risks = np.concatenate(all_train_risks)
    all_train_times = np.concatenate(all_train_times)
    all_train_events = np.concatenate(all_train_events)

    # Fit baseline hazard using training data
    train_df = pd.DataFrame({
        'duration': all_train_times,
        'event': all_train_events,
        'risk': all_train_risks
    })
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='duration', event_col='event')
    baseline_hazard_df = cph.baseline_cumulative_hazard_

    def baseline_cumulative_hazard(t):
        # For discrete time points, lifelines index may not contain every time explicitly.
        # You might want to interpolate. For simplicity, do what your first function does:
        if t in baseline_hazard_df.index:
            return baseline_hazard_df.loc[t].values[0]
        else:
            # If t is beyond the last row, take the last value
            return baseline_hazard_df.values[-1, 0]

    # 3) Now compute IBS on the validation set *using the best model*
    #    We re-predict risks on the validation set, then do Brier scoring with IPCW, as in your snippet.
    risks_list, times_list, events_list = [], [], []
    with torch.no_grad():
        for batch in unified_val_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = (
                x_cat.to(device),
                x_num.to(device),
                x_wsi.to(device),
                mask_wsi.to(device)
            )
            risk = late_fusion_model(x_cat, x_num, x_wsi, mask_wsi)
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            risks_list.append(risk.cpu().numpy())
            times_list.append(t_b.numpy())
            events_list.append(e_b.numpy())

    val_risks = np.concatenate(risks_list)
    val_times = np.concatenate(times_list)
    val_events = np.concatenate(events_list).astype(bool)

      # Step 1: Prepare data
    risk_scores = val_risks  # Your risk scores
    times = val_times  # Event times
    events = val_events  # Event indicators (1 = event, 0 = censored)

    risk_scores = np.clip(risk_scores, -4, 4)  # First: control exp(risk)  
    # Step 2: Estimate baseline survival function
    baseline_surv = cph.baseline_survival_
    time_points = baseline_surv.index.values
    baseline_survival_at_times = baseline_surv['baseline survival'].values
    # Step 3: Compute time-dependent survival probabilities
    time_dependent_survival = np.array([
        baseline_survival_at_times ** np.exp(score) for score in risk_scores
    ])

    # Step 4: Create a DataFrame of survival probabilities
    # Each row corresponds to a patient, and each column corresponds to a time point
    survival_df = pd.DataFrame(time_dependent_survival, columns=time_points)

    #survival_df = survival_df.clip(lower=0.0, upper=1.0)
    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  
    survival_df = survival_df.apply(
        lambda row: pd.Series(np.minimum.accumulate(row.values), index=row.index),
        axis=1
    )

    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  

    # Step 5: Remove the last quartile of time points
    time_grid = time_points[time_points <= np.quantile(times[events == 1], 0.75)]

    survival_df.columns = survival_df.columns.astype(np.float32)
    time_grid = time_grid.astype(np.float32)

    # Step 6: Compute IBS using EvalSurv
    # Transpose survival_df to match EvalSurv's expected format
    ev = EvalSurv(survival_df.T, times, events, censor_surv='km')
    ibs = min(ev.integrated_brier_score(time_grid), 0.25)  # Compute IBS
    best_ibs = ibs

    print(f"[Late Fusion] Best Val C-index was {best_cindex:.4f}; IBS (computed at end) = {ibs:.4f}")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    return best_cindex, ibs












'''

def train_evaluate_fold_clinical(params, train_loader, val_loader, categories, num_continuous, device='cuda'):
    """
    Trains and evaluates the SurvivalModel_clinical on one fold using:
      - Pycox's CoxPHLoss
      - scikit-survival's concordance_index_censored

    Args:
        params (dict): Hyperparameters, e.g.:
            {
                "lr": 1e-4,
                "epochs": 20,
                "dim": 128,
                "depth": 6,
                "heads": 8,
                "attn_dropout": 0.3,
                "ff_dropout": 0.3,
                "batch_size": 64  # typically used in data loader
            }
        train_loader (DataLoader): Yields (X_cat_batch, X_num_batch, t_batch, e_batch).
        val_loader (DataLoader):   Yields (X_cat_batch, X_num_batch, t_batch, e_batch).
        categories (list[int]): Cardinalities for each categorical feature.
        num_continuous (int):     Number of continuous features.
        device (str):             'cuda' or 'cpu'.

    Returns:
        float: C-index on the validation set after final epoch.
    """

    # Unpack hyperparameters from params (use defaults if any missing).
    lr = params.get("lr", 1e-4)
    epochs = params.get("epochs", 20)
    dim_val = params.get("dim", 64)
    depth_val = params.get("depth", 6)
    heads_val = params.get("heads", 8)
    attn_drop_val = params.get("attn_dropout", 0.3)
    ff_drop_val = params.get("ff_dropout", 0.3)

    # ---------------------------------------------------------
    # If you want to automatically apply the batch size as well:
    # batch_size = params.get("batch_size", 64)
    # # Then you would reconstruct train_loader and val_loader inside here
    # # but typically you'd do that externally, so you have the data ready.
    # ---------------------------------------------------------

    # --- 1. Define the model --- #
    model = SurvivalModel_clinical(
        categories=categories,
        num_continuous=num_continuous,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # --- 2. Define the optimizer & loss --- #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CoxPHLoss()

    best_cindex = 0  # Track the best C-index

    # --- 3. Training loop --- #
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_cat_batch, X_num_batch, t_batch, e_batch in train_loader:
            X_cat_batch = X_cat_batch.to(device)
            X_num_batch = X_num_batch.to(device)
            t_batch = t_batch.to(device)
            e_batch = e_batch.to(device)

            optimizer.zero_grad()
            risk_scores = model(X_cat_batch, X_num_batch)
            loss = loss_fn(risk_scores, t_batch, e_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # --- Compute baseline cumulative hazard H_0(t) after each epoch --- #
            all_train_risks, all_train_times, all_train_events = [], [], []

            with torch.no_grad():
                for X_cat_batch, X_num_batch, t_batch, e_batch in train_loader:
                    X_cat_batch = X_cat_batch.to(device)
                    X_num_batch = X_num_batch.to(device)
                    risk_scores = model(X_cat_batch, X_num_batch).cpu().numpy()


                    all_train_risks.append(risk_scores)
                    all_train_times.append(t_batch.cpu().numpy())
                    all_train_events.append(e_batch.cpu().numpy())

            all_train_risks = np.concatenate(all_train_risks)
            all_train_times = np.concatenate(all_train_times)
            all_train_events = np.concatenate(all_train_events)

            # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
            from lifelines import CoxPHFitter
            train_df = pd.DataFrame({
                'duration': all_train_times,
                'event': all_train_events,
                'risk': all_train_risks
            })
            cph = CoxPHFitter()
            cph.fit(train_df, duration_col='duration', event_col='event')
            baseline_hazard_df = cph.baseline_cumulative_hazard_

            def baseline_cumulative_hazard(t):
                if t in baseline_hazard_df.index:
                    return baseline_hazard_df.loc[t].values[0]
                else:
                    return baseline_hazard_df.values[-1, 0]



        # Compute validation C-index after each epoch
        model.eval()
        all_risks, all_times, all_events = [], [], []


        with torch.no_grad():
            for X_cat_batch, X_num_batch, t_batch, e_batch in val_loader:
                X_cat_batch = X_cat_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                risk_scores = model(X_cat_batch, X_num_batch).cpu()

                all_risks.append(risk_scores)
                all_times.append(t_batch)
                all_events.append(e_batch)

        # Concatenate tensors
        all_risks = torch.cat(all_risks)
        all_times = torch.cat(all_times)
        all_events = torch.cat(all_events)

        # Convert to numpy
        risks_val = all_risks.numpy()
        times_val = all_times.numpy()
        events_val = all_events.numpy().astype(bool)




        # --- Compute IPCW weights for censored patients --- #
        kmf = KaplanMeierFitter()
        kmf.fit(times_val, event_observed=events_val == 0)  # Only censored patients

        time_points = np.sort(np.unique(times_val))  # Use actual event times
        ipcw_weights = {}
        
        n = len(times_val)  # Sample size for truncation formula
        lower_bound = 5 / (np.sqrt(n) * np.log(n))  # Sample-size-based lower bound
        for t in time_points:
            G_t = kmf.survival_function_at_times(t).values[0]
            #print("G IS ",G_t)
            G_t = max(G_t,lower_bound)

            ipcw_weights[t] = 1 / np.clip(G_t, 1e-8, None)  # Avoid division by zero

        # --- Compute IBS directly --- #
        ibs_values = []

        for t in time_points:
            H_t = baseline_cumulative_hazard(t)  # Get H_0(t)
            #S_t = np.exp(-H_t * risks_val)  # Convert risk scores to survival probabilities
            S_t = np.exp(-H_t)  # DO NOT multiply again by risks_val
            #print(f"Epoch {epoch}: S_t min={S_t.min()}, max={S_t.max()}")
        

            # Apply IPCW only to censored patients
            censor_mask = (events_val == 0)  # Mask for censored patients
            brier_t = np.mean(
                (S_t - (times_val > t)) ** 2 * np.where(censor_mask, ipcw_weights[t], 1)
            )
            ibs_values.append(brier_t)

        # Compute final IBS as the mean of all Brier Scores across event times
        #print(ibs_values)
        ibs = np.mean(ibs_values)

        #print(ibs)

        # Compute the C-index for this epoch
        cindex = concordance_index_censored(
            event_indicator=events_val,
            event_time=times_val,
            estimate=risks_val
        )[0]

        # --- Track Best C-index and corresponding IBS Per Epoch --- #
        if epoch == 0:
            best_cindex = cindex
            best_ibs_at_best_cindex = ibs
            best_epoch = epoch
        else:
            if cindex > best_cindex:
                best_cindex = cindex
                best_ibs_at_best_cindex = ibs
                best_ibs_epoch = epoch
                
        # Update the best C-index
        #if cindex > best_cindex:
        #    best_cindex = cindex

        # Print loss, validation C-index, and IBS every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Validation C-Index: {cindex:.4f} (Best: {best_cindex:.4f}) | IBS: {ibs:.4f} (IBS at Best C-Index: {best_ibs_at_best_cindex:.4f})")

    print(f"\nBest Validation C-Index: {best_cindex:.4f}")
    print(f"Corresponding IBS: {best_ibs_at_best_cindex:.4f}")

    return best_cindex,best_ibs_at_best_cindex
'''
def train_evaluate_fold_clinical(params, train_loader, val_loader, categories, num_continuous, save_dir, device='cuda'):
    """
    Trains and evaluates the SurvivalModel_clinical on one fold using:
      - Pycox's CoxPHLoss
      - scikit-survival's concordance_index_censored

    Args:
        params (dict): Hyperparameters, e.g.:
            {
                "lr": 1e-4,
                "epochs": 20,
                "dim": 128,
                "depth": 6,
                "heads": 8,
                "attn_dropout": 0.3,
                "ff_dropout": 0.3,
                "batch_size": 64  # typically used in data loader
            }
        train_loader (DataLoader): Yields (X_cat_batch, X_num_batch, t_batch, e_batch).
        val_loader (DataLoader):   Yields (X_cat_batch, X_num_batch, t_batch, e_batch).
        categories (list[int]): Cardinalities for each categorical feature.
        num_continuous (int):     Number of continuous features.
        device (str):             'cuda' or 'cpu'.

    Returns:
        float: C-index on the validation set after final epoch.
    """

    # Unpack hyperparameters from params (use defaults if any missing).
    lr = params.get("lr", 1e-4)
    epochs = params.get("epochs", 20)
    dim_val = params.get("dim", 64)
    depth_val = params.get("depth", 6)
    heads_val = params.get("heads", 8)
    attn_drop_val = params.get("attn_dropout", 0.3)
    ff_drop_val = params.get("ff_dropout", 0.3)

    # ---------------------------------------------------------
    # If you want to automatically apply the batch size as well:
    # batch_size = params.get("batch_size", 64)
    # # Then you would reconstruct train_loader and val_loader inside here
    # # but typically you'd do that externally, so you have the data ready.
    # ---------------------------------------------------------

    # --- 1. Define the model --- #
    model = SurvivalModel_clinical(
        categories=categories,
        num_continuous=num_continuous,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # --- 2. Define the optimizer & loss --- #
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CoxPHLoss()

    best_cindex = 0  # Track the best C-index
    best_epoch = 0

    # --- 3. Training loop --- #
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_cat_batch, X_num_batch, t_batch, e_batch in train_loader:
            if e_batch.sum() == 0:
                print("Skipping batch with all censoredpythoin cases")
                continue  # Skip this batch
            X_cat_batch = X_cat_batch.to(device)
            X_num_batch = X_num_batch.to(device)
            t_batch = t_batch.to(device)
            e_batch = e_batch.to(device)
            optimizer.zero_grad()
            risk_scores = model(X_cat_batch, X_num_batch)
            loss = loss_fn(risk_scores, t_batch, e_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        # Compute validation C-index after each epoch
        model.eval()
        all_risks, all_times, all_events = [], [], []


        with torch.no_grad():
            for X_cat_batch, X_num_batch, t_batch, e_batch in val_loader:
                X_cat_batch = X_cat_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                risk_scores = model(X_cat_batch, X_num_batch).cpu()

                all_risks.append(risk_scores)
                all_times.append(t_batch)
                all_events.append(e_batch)

        # Concatenate tensors
        all_risks = torch.cat(all_risks)
        all_times = torch.cat(all_times)
        all_events = torch.cat(all_events)

        # Convert to numpy
        risks_val = all_risks.numpy()
        times_val = all_times.numpy()
        events_val = all_events.numpy().astype(bool)


        # Compute the C-index for this epoch
        cindex = concordance_index_censored(
            event_indicator=events_val,
            event_time=times_val,
            estimate=risks_val
        )[0]

        # --- Track Best C-index and corresponding IBS Per Epoch --- #

        if cindex>best_cindex:
            if 'best_model_path' in locals() and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_cindex=cindex
            su1=uuid.uuid4().hex[:6]
            su2=uuid.uuid4().hex[:6]
            best_model_path=os.path.join(save_dir,f"best_model_epoch_{epoch}_{su1}_{su2}.pth")
            best_epoch = epoch
            torch.save(model.state_dict(),best_model_path)

                
        # Update the best C-index
        #if cindex > best_cindex:
        #    best_cindex = cindex

        # Print loss, validation C-index, and IBS every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Validation C-Index: {cindex:.4f} (Best: {best_cindex:.4f})")

 
    print("best epoch:", best_epoch)
    # Compute IBS only for the best model
    model.load_state_dict(torch.load(best_model_path))
    all_train_risks, all_train_times, all_train_events = [], [], []

    with torch.no_grad():
        for X_cat_batch, X_num_batch, t_batch, e_batch in train_loader:

            X_cat_batch = X_cat_batch.to(device)
            X_num_batch = X_num_batch.to(device)
            risk_scores = model(X_cat_batch, X_num_batch).cpu().numpy()


            all_train_risks.append(risk_scores)
            all_train_times.append(t_batch.cpu().numpy())
            all_train_events.append(e_batch.cpu().numpy())

    all_train_risks = np.concatenate(all_train_risks)
    all_train_times = np.concatenate(all_train_times)
    all_train_events = np.concatenate(all_train_events)

    # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
    from lifelines import CoxPHFitter
    train_df = pd.DataFrame({
        'duration': all_train_times,
        'event': all_train_events,
        'risk': all_train_risks
    })
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='duration', event_col='event')
    baseline_hazard_df = cph.baseline_cumulative_hazard_

    def baseline_cumulative_hazard(t):
        if t in baseline_hazard_df.index:
            return baseline_hazard_df.loc[t].values[0]
        else:
            return baseline_hazard_df.values[-1, 0]



    model.eval()
    all_risks, all_times, all_events = [], [], []


    with torch.no_grad():
        for X_cat_batch, X_num_batch, t_batch, e_batch in val_loader:
            X_cat_batch = X_cat_batch.to(device)
            X_num_batch = X_num_batch.to(device)
            risk_scores = model(X_cat_batch, X_num_batch).cpu()

            all_risks.append(risk_scores)
            all_times.append(t_batch)
            all_events.append(e_batch)

    # Concatenate tensors
    all_risks = torch.cat(all_risks)
    all_times = torch.cat(all_times)
    all_events = torch.cat(all_events)

    # Convert to numpy
    risks_val = all_risks.numpy()
    times_val = all_times.numpy()
    events_val = all_events.numpy().astype(bool)




    # Step 1: Prepare data
    risk_scores = risks_val  # Your risk scores
    times = times_val  # Event times
    events = events_val  # Event indicators (1 = event, 0 = censored)

    risk_scores = np.clip(risk_scores, -4, 4)  # First: control exp(risk)  
    # Step 2: Estimate baseline survival function
    baseline_surv = cph.baseline_survival_
    time_points = baseline_surv.index.values
    baseline_survival_at_times = baseline_surv['baseline survival'].values
    # Step 3: Compute time-dependent survival probabilities
    time_dependent_survival = np.array([
        baseline_survival_at_times ** np.exp(score) for score in risk_scores
    ])

    # Step 4: Create a DataFrame of survival probabilities
    # Each row corresponds to a patient, and each column corresponds to a time point
    survival_df = pd.DataFrame(time_dependent_survival, columns=time_points)

    #survival_df = survival_df.clip(lower=0.0, upper=1.0)
    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  
    survival_df = survival_df.apply(
        lambda row: pd.Series(np.minimum.accumulate(row.values), index=row.index),
        axis=1
    )

    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  

    # Step 5: Remove the last quartile of time points
    time_grid = time_points[time_points <= np.quantile(times[events == 1], 0.75)]

    survival_df.columns = survival_df.columns.astype(np.float32)
    time_grid = time_grid.astype(np.float32)

    # Step 6: Compute IBS using EvalSurv
    # Transpose survival_df to match EvalSurv's expected format
    ev = EvalSurv(survival_df.T, times, events, censor_surv='km')
    ibs = min(ev.integrated_brier_score(time_grid), 0.25)  # Compute IBS
    best_ibs = ibs


    print(f"\nBest Validation C-Index: {best_cindex:.4f}")
    print(f"Corresponding IBS: {ibs:.4f}")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    return best_cindex,ibs

def train_evaluate_early_attention_fusion(
    clinical_train,
    clinical_val,
    ds_orig_train,
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    device='cuda'
):
    """
    Toy function that unrolls:
      1) Clinical survival model training/validation
      2) WSI survival model training/validation
    Returns:
      (clinical_cindex, wsi_cindex)
    """

    ###############################################################################
    # PART A: CLINICAL TRAINING AND VALIDATION
    ###############################################################################
    # 1) Standardize numeric features
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val   = scaler.transform(clinical_val[numeric_cols])

    # Rebuild numeric/cat dataframes
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num   = pd.DataFrame(scaled_val,   columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat   = clinical_val[categorical_cols]

    # 2) Convert to Tensors
    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat   = torch.tensor(val_cat.values,   dtype=torch.long)
    X_val_num   = torch.tensor(val_num.values,   dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train= clinical_train["PFI"].values.astype(bool)
    time_val   = clinical_val["PFI.time"].values.astype(float)
    event_val  = clinical_val["PFI"].values.astype(bool)

    train_dataset_clin = torch.utils.data.TensorDataset(
        X_train_cat, X_train_num,
        torch.FloatTensor(time_train),
        torch.FloatTensor(event_train)
    )
    val_dataset_clin = torch.utils.data.TensorDataset(
        X_val_cat, X_val_num,
        torch.FloatTensor(time_val),
        torch.FloatTensor(event_val)
    )

    train_loader_clin = DataLoader(train_dataset_clin, batch_size=clin_param.get('batch_size', 64), shuffle=True)
    val_loader_clin   = DataLoader(val_dataset_clin,   batch_size=clin_param.get('batch_size', 64), shuffle=False)

    # 3) Define Clinical Model
    from torch import nn
    cat_cardinalities = [
        int(clinical_common[col].max() + 1)
        for col in categorical_cols if col in clinical_common.columns
    ]
    num_cont = len(numeric_cols)

    # Build the FT-Transformer-based clinical model
    # (SurvivalModel_clinical is presumably your existing class)
    lr            = clin_param.get("lr", 1e-4)
    epochs        = clin_param.get("epochs", 20)
    dim_val       = clin_param.get("dim", 64)
    depth_val     = clin_param.get("depth", 6)
    heads_val     = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val   = clin_param.get("ff_dropout", 0.3)

    from pycox.models.loss import CoxPHLoss

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    optimizer_clin = torch.optim.Adam(clinical_model.parameters(), lr=lr)
    loss_fn_clin   = CoxPHLoss()

    # 4) Train Clinical Model
    for epoch in range(epochs):
        clinical_model.train()
        total_loss = 0.0
        for X_cat_b, X_num_b, t_b, e_b in train_loader_clin:
            X_cat_b, X_num_b = X_cat_b.to(device), X_num_b.to(device)
            t_b, e_b = t_b.to(device), e_b.to(device)

            optimizer_clin.zero_grad()
            risk_scores = clinical_model(X_cat_b, X_num_b)
            loss = loss_fn_clin(risk_scores, t_b, e_b)
            loss.backward()
            optimizer_clin.step()
            total_loss += loss.item()

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader_clin)
            print(f"[Fusion: Clinical] Epoch {epoch+1}/{epochs}, CoxPHLoss: {avg_loss:.4f}")

    # 5) Validate Clinical Model
    clinical_model.eval()
    all_risks_clin, all_times_clin, all_events_clin = [], [], []
    with torch.no_grad():
        for X_cat_b, X_num_b, t_b, e_b in val_loader_clin:
            X_cat_b, X_num_b = X_cat_b.to(device), X_num_b.to(device)
            risk_scores = clinical_model(X_cat_b, X_num_b).cpu()
            all_risks_clin.append(risk_scores)
            all_times_clin.append(t_b)
            all_events_clin.append(e_b)

    all_risks_clin  = torch.cat(all_risks_clin).numpy()
    all_times_clin  = torch.cat(all_times_clin).numpy()
    all_events_clin = torch.cat(all_events_clin).numpy().astype(bool)

    # Typically negative is used if higher risk => earlier event
    from lifelines.utils import concordance_index
    clinical_cindex = concordance_index_censored(
        event_indicator=all_events_clin,
        event_time=all_times_clin,
        estimate=all_risks_clin
    )[0]
    print(f"[Fusion: Clinical] Validation C-Index: {clinical_cindex:.4f}")

    ###############################################################################
    # PART B: WSI TRAINING AND VALIDATION
    ###############################################################################
    # ds_orig_train, ds_orig_val are Subsets of your WSI dataset
    # Each item from ds_orig_train returns (region_tensor, time, event).
    # We use your existing code for train_evaluate_wsi, but unrolled here

    # Build DataLoaders
    wsi_batch_size = wsi_param.get('batch_size', 64)
    train_loader_wsi = DataLoader(ds_orig_train, batch_size=wsi_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader_wsi   = DataLoader(ds_orig_val,   batch_size=wsi_batch_size, shuffle=False, collate_fn=collate_fn)

    # Create WSI model (SurvivalModel) 
    # using your param keys: hidden_dim, dropout, weight_decay, epochs, grad_clip, etc.
    wsi_lr         = wsi_param.get('lr', 1e-3)
    wsi_dropout    = wsi_param.get('dropout', 0.0)
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_wd         = wsi_param.get('weight_decay', 1e-5)
    wsi_epochs     = wsi_param.get('epochs', 50)
    wsi_grad_clip  = wsi_param.get('grad_clip', 1.0)

    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    optimizer_wsi = torch.optim.Adam(wsi_model.parameters(), lr=wsi_lr, weight_decay=wsi_wd)
    loss_fn_wsi   = CoxPHLoss()

    best_cindex_wsi = 0  # if you want to keep track of the best C-index

    # Training loop
    for epoch in range(wsi_epochs):
        wsi_model.train()
        for x_b, mask_b, t_b, e_b in train_loader_wsi:
            x_b    = x_b.to(device)
            mask_b = mask_b.to(device)
            t_b    = t_b.to(device)
            e_b    = e_b.to(device)

            optimizer_wsi.zero_grad()
            risk_scores = wsi_model(x_b, mask_b)
            loss = loss_fn_wsi(risk_scores, t_b, e_b)
            loss.backward()

            if wsi_grad_clip:
                torch.nn.utils.clip_grad_norm_(wsi_model.parameters(), wsi_grad_clip)
            optimizer_wsi.step()

        # Evaluate after each epoch
        wsi_model.eval()
        risks_list, times_list, events_list = [], [], []
        with torch.no_grad():
            for x_b, mask_b, t_b, e_b in val_loader_wsi:
                x_b, mask_b = x_b.to(device), mask_b.to(device)
                r = wsi_model(x_b, mask_b)
                risks_list.append(r.cpu())
                times_list.append(t_b)
                events_list.append(e_b)
        # Collate
        final_risks = torch.cat(risks_list).cpu().numpy()
        final_times = torch.cat(times_list).cpu().numpy()
        final_events= torch.cat(events_list).cpu().numpy()
        # negative because higher risk => earlier event
        current_cindex = concordance_index(final_times, -final_risks, final_events)
        best_cindex_wsi = max(best_cindex_wsi, current_cindex)

        if (epoch+1) % 5 == 0:
            print(f"[Fusion: WSI] Epoch {epoch+1}/{wsi_epochs}, Current Val C-index: {current_cindex:.4f}")

    wsi_cindex = best_cindex_wsi
    print(f"[Fusion: WSI] Best Validation C-Index: {wsi_cindex:.4f}")

    ###############################################################################
    # RETURN BOTH
    ###############################################################################
    return clinical_cindex, wsi_cindex


#############################################
# Fusion Modules
#############################################

class CrossModalAttentionFusion(nn.Module):
    """
    Fuses clinical and WSI embeddings (each (B, 192)) using multi-head self-attention.
    """
    def __init__(self, embed_dim=192, num_heads=2):
        super(CrossModalAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    def forward(self, h_clin, h_wsi):
        # Stack and transpose to shape (2, B, 192)
        h_seq = torch.stack([h_clin, h_wsi], dim=1).transpose(0, 1)
        attn_output, _ = self.attn(h_seq, h_seq, h_seq)  # (2, B, 192)
        h_fused = attn_output.mean(dim=0)                # (B, 192)
        return h_fused


class ConcatenationFusion(nn.Module):
    """
    Concatenates clinical and WSI embeddings (each (B, 192)) into a single (B, 384) representation.
    """
    def __init__(self):
        super(ConcatenationFusion, self).__init__()

    def forward(self, h_clin, h_wsi):
        # Concatenate along the feature dimension
        h_fused = torch.cat([h_clin, h_wsi], dim=-1)  # (B, 384)
        return h_fused


class FinalSurvivalHead(nn.Module):
    """
    Maps the fused embedding (B, 192) to a single risk score.
    """
    def __init__(self, input_dim=192):
        super(FinalSurvivalHead, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)

#############################################
# Unified Model and Dataset
#############################################

class UnifiedSurvivalModel(nn.Module):
    """
    Unified model that extracts clinical and WSI embeddings,
    fuses them via cross-modal attention, and outputs a risk score.
    """
    def __init__(self, clinical_model, wsi_model, fusion_module, final_head):
        super(UnifiedSurvivalModel, self).__init__()
        self.clinical_model = clinical_model
        self.wsi_model = wsi_model
        self.fusion_module = fusion_module
        self.final_head = final_head
    
    def forward(self, x_cat, x_num, x_wsi, mask_wsi):
        # Extract clinical embedding (B, 192)
        h_clin = self.clinical_model.ft_transformer(x_cat, x_num, return_embedding=True)
        # Extract WSI embedding using the attention aggregator
        h_wsi = self.wsi_model.aggregator(x_wsi, mask_wsi)
        # Fuse embeddings via cross-modal attention
        h_fused = self.fusion_module(h_clin, h_wsi)
        # Compute risk score
        risk = self.final_head(h_fused)
        return risk

class UnifiedDataset(Dataset):
    """
    Combines clinical and WSI data for aligned patients.
    Assumes clinical data tensors are aligned with wsi_dataset order.
    Each item returns: (x_cat, x_num, x_wsi, mask_wsi, time, event)
    """
    def __init__(self, clin_cat, clin_num, times, events, wsi_dataset):
        self.clin_cat = clin_cat
        self.clin_num = clin_num
        self.times = times
        self.events = events
        self.wsi_dataset = wsi_dataset  # Each sample returns (regions, time, event)
    
    def __len__(self):
        return len(self.clin_cat)
    
    def __getitem__(self, idx):
        x_cat = self.clin_cat[idx]
        x_num = self.clin_num[idx]
        time = self.times[idx]
        event = self.events[idx]
        # Retrieve raw WSI embedding (regions) from dataset; note: it returns (regions, time, event)
        regions, _, _ = self.wsi_dataset[idx]
        # Create a mask assuming all patches are valid
        mask = torch.ones(regions.shape[0], dtype=torch.bool)
        return x_cat, x_num, regions, mask, time, event

def unified_collate_fn(batch):
    """
    Pads variable-length WSI embeddings and masks.
    Returns:
      (x_cat, x_num, x_wsi_padded, mask_padded, times, events)
    """
    x_cat, x_num, x_wsi, masks, times, events = zip(*batch)
    x_cat = torch.stack(x_cat)
    x_num = torch.stack(x_num)
    x_wsi_padded = pad_sequence(x_wsi, batch_first=True, padding_value=0.0)  # (B, max_N, 192)
    mask_padded = pad_sequence(masks, batch_first=True, padding_value=False)  # (B, max_N)
    times = torch.tensor(times, dtype=torch.float32)
    events = torch.tensor(events, dtype=torch.float32)
    return x_cat, x_num, x_wsi_padded, mask_padded, times, events



# ----- VAE Fusion Module -----
class VAEFusion(nn.Module):
    """
    VAE Fusion module that jointly encodes concatenated clinical (192D)
    and WSI (192D) embeddings into a latent space and decodes them back.
    Input: (B, 384)
    Outputs:
      - z: latent representation (B, latent_dim)
      - x_recon: reconstruction of the input (B, 384)
      - mu: encoder mean (B, latent_dim)
      - logvar: encoder log-variance (B, latent_dim)
    """
    def __init__(self, input_dim=384, latent_dim=192, hidden_dim=256):
        super(VAEFusion, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)            # (B, hidden_dim)
        mu = self.fc_mu(h)             # (B, latent_dim)
        logvar = self.fc_logvar(h)     # (B, latent_dim)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)      # (B, input_dim)
        return z, x_recon, mu, logvar


# ----- Unified Model with VAE Fusion -----
class UnifiedSurvivalModelVAE(nn.Module):
    """
    Unified model that:
      - Extracts clinical embeddings via FTTransformer from clinical inputs.
      - Extracts WSI embeddings via an attention aggregator from WSI inputs.
      - Concatenates these to form a 384D vector per patient.
      - Passes the concatenated vector through a VAE fusion module to obtain a joint latent representation.
      - Maps the latent representation to a risk score via a final survival head.
    """
    def __init__(self, clinical_model, wsi_model, vae_fusion, final_head):
        super(UnifiedSurvivalModelVAE, self).__init__()
        self.clinical_model = clinical_model  # Expected to have an ft_transformer with return_embedding=True
        self.wsi_model = wsi_model            # Expected to have an aggregator method that accepts (x_wsi, mask)
        self.vae_fusion = vae_fusion
        self.final_head = final_head
    
    def forward(self, x_cat, x_num, x_wsi, mask_wsi):
        # Extract clinical embedding: (B, 192)
        h_clin = self.clinical_model.ft_transformer(x_cat, x_num, return_embedding=True)
        # Extract WSI embedding: (B, 192)
        h_wsi = self.wsi_model.aggregator(x_wsi, mask_wsi)
        # Concatenate to form joint input: (B, 384)
        cat_emb = torch.cat([h_clin, h_wsi], dim=1)
        # Pass through VAE fusion module
        z, x_recon, mu, logvar = self.vae_fusion(cat_emb)
        # Map latent representation to risk score
        risk = self.final_head(z)
        # Also return the original concatenated embedding for reconstruction loss
        return risk, x_recon, mu, logvar, cat_emb






#############################################
# Unified End-to-End Training Function
#############################################
'''
def train_evaluate_end_to_end_fusion(
    clinical_train,
    clinical_val,
    ds_orig_train,  # Subset of your WSI dataset; each sample returns (regions, time, event)
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    device='cuda:1'
):
    """
    End-to-end training of the unified multimodal survival model.
    Processes clinical inputs and WSI inputs, aligns them via a unified dataset,
    instantiates the clinical branch, WSI branch, fusion module, and final survival head,
    and trains the network with a single CoxPHLoss.
    Returns the validation concordance index.
    """
    # --- Process Clinical Data ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val = scaler.transform(clinical_val[numeric_cols])
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat = clinical_val[categorical_cols]

    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat = torch.tensor(val_cat.values, dtype=torch.long)
    X_val_num = torch.tensor(val_num.values, dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train = clinical_train["PFI"].values.astype(bool)
    time_val = clinical_val["PFI.time"].values.astype(float)
    event_val = clinical_val["PFI"].values.astype(bool)

    # --- Build Unified Datasets ---
    unified_train_dataset = UnifiedDataset(
        clin_cat=X_train_cat,
        clin_num=X_train_num,
        times=time_train,
        events=event_train,
        wsi_dataset=ds_orig_train
    )
    unified_val_dataset = UnifiedDataset(
        clin_cat=X_val_cat,
        clin_num=X_val_num,
        times=time_val,
        events=event_val,
        wsi_dataset=ds_orig_val
    )
    batch_size = clin_param.get('batch_size', 64)
    unified_train_loader = DataLoader(unified_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=unified_collate_fn)
    unified_val_loader = DataLoader(unified_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)

    # --- Instantiate Branches & Fusion Modules ---
    cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]
    num_cont = len(numeric_cols)

    # Clinical branch parameters
    lr_clin = clin_param.get("lr", 1e-4)
    epochs_clin = clin_param.get("epochs", 20)
    dim_val = clin_param.get("dim", 64)
    depth_val = clin_param.get("depth", 6)
    heads_val = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val = clin_param.get("ff_dropout", 0.3)

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # WSI branch parameters
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_dropout = wsi_param.get('dropout', 0.0)
    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    # Fusion and final head
    #fusion_module = CrossModalAttentionFusion(embed_dim=192, num_heads=2).to(device)
    #final_surv_head = FinalSurvivalHead(input_dim=192).to(device)
    fusion_module = ConcatenationFusion().to(device)
    final_surv_head = FinalSurvivalHead(input_dim=384).to(device)
    

    unified_model = UnifiedSurvivalModel(
        clinical_model=clinical_model,
        wsi_model=wsi_model,
        fusion_module=fusion_module,
        final_head=final_surv_head
    ).to(device)

    # --- Define Optimizer & Loss ---
    optimizer = torch.optim.Adam(unified_model.parameters(), lr=lr_clin)
    loss_fn = CoxPHLoss()

    # ----- End-to-End Training Loop -----
    #unified_epochs = max(epochs_clin, wsi_param.get('epochs',50),50)
    unified_epochs = 70
    best_cindex = 0
    best_ibs_at_best_cindex = 0
    for epoch in range(unified_epochs):
        unified_model.train()
        total_loss = 0.0
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
            t_b, e_b = t_b.to(device), e_b.to(device)
            optimizer.zero_grad()
            risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            loss = loss_fn(risk, t_b, e_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # --- Compute baseline cumulative hazard H_0(t) after each epoch --- #
            all_train_risks, all_train_times, all_train_events = [], [], []

            with torch.no_grad():
                for batch in unified_train_loader:
                    x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                    x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)

                    risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)

                    all_train_risks.append(risk.cpu().numpy())
                    all_train_times.append(t_b.cpu().numpy())
                    all_train_events.append(e_b.cpu().numpy())

            all_train_risks = np.concatenate(all_train_risks)
            all_train_times = np.concatenate(all_train_times)
            all_train_events = np.concatenate(all_train_events)

            # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
            from lifelines import CoxPHFitter
            train_df = pd.DataFrame({
                'duration': all_train_times,
                'event': all_train_events,
                'risk': all_train_risks
            })
            cph = CoxPHFitter()
            cph.fit(train_df, duration_col='duration', event_col='event')
            baseline_hazard_df = cph.baseline_cumulative_hazard_

            def baseline_cumulative_hazard(t):
                if t in baseline_hazard_df.index:
                    return baseline_hazard_df.loc[t].values[0]
                else:
                    return baseline_hazard_df.values[-1, 0]



        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(unified_train_loader)
            # ----- Validation after each epoch -----
            unified_model.eval()
            risks_list, times_list, events_list = [], [], []
            with torch.no_grad():
                for batch in unified_val_loader:
                    x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                    x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
                    risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
                    risks_list.append(risk.cpu())
                    times_list.append(t_b)
                    events_list.append(e_b)
            final_risks = torch.cat(risks_list).cpu().numpy()
            final_times = torch.cat(times_list).cpu().numpy()
            final_events = torch.cat(events_list).cpu().numpy()

            # Convert to numpy
            risks_val = final_risks
            times_val = final_times
            events_val = final_events.astype(bool)


            # --- Compute IPCW weights for censored patients --- #
            kmf = KaplanMeierFitter()
            kmf.fit(times_val, event_observed=events_val == 0)  # Only censored patients

            time_points = np.sort(np.unique(times_val))  # Use actual event times
            ipcw_weights = {}
            n = len(times_val)  # Sample size for truncation formula
            lower_bound = 5 / (np.sqrt(n) * np.log(n))  # Sample-size-based lower bound

            for t in time_points:
                G_t = kmf.survival_function_at_times(t).values[0]
                G_t = max(G_t, lower_bound)
                ipcw_weights[t] = 1 / np.clip(G_t, 1e-8, None)  # Avoid division by zero

            # --- Compute IBS directly --- #
            ibs_values = []

            for t in time_points:
                H_t = baseline_cumulative_hazard(t)  # Get H_0(t)
                #S_t = np.exp(-H_t * risks_val)  # Convert risk scores to survival probabilities
                S_t = np.exp(-H_t)  # DO NOT multiply again by risks_val
                #print(f"Epoch {epoch}: S_t min={S_t.min()}, max={S_t.max()}")
            

                # Apply IPCW only to censored patients
                censor_mask = (events_val == 0)  # Mask for censored patients
                brier_t = np.mean(
                    (S_t - (times_val > t)) ** 2 * np.where(censor_mask, ipcw_weights[t], 1)
                )
                ibs_values.append(brier_t)

            # Compute final IBS as the mean of all Brier Scores across event times
            #print(ibs_values)
            ibs = np.mean(ibs_values)

            cindex = concordance_index_censored(
                event_indicator=events_val,
                event_time=times_val,
                estimate=risks_val
            )[0]

            if cindex > best_cindex:
                best_cindex = cindex
                best_ibs_at_best_cindex = ibs


            print(f"[Unified Fusion] Epoch {epoch+1}/{unified_epochs}, Loss: {avg_loss:.4f}, Current Val C-index: {cindex:.4f} | IBS: {ibs:.4f} (IBS at Best C-Index: {best_ibs_at_best_cindex:.4f})")
    print(f"[Unified Fusion] Best Validation C-Index: {best_cindex:.4f}")
    return best_cindex, best_ibs_at_best_cindex
'''
def train_evaluate_end_to_end_fusion(
    cp_model,
    clinical_train,
    clinical_val,
    ds_orig_train,
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    save_dir="saved_models",
    device='cuda:1'
):
    """
    End-to-end training of the unified multimodal survival model.
    Processes clinical inputs and WSI inputs, aligns them via a unified dataset,
    instantiates the clinical branch, WSI branch, fusion module, and final survival head,
    and trains the network with a single CoxPHLoss.
    Returns the validation concordance index.
    """
    # --- Process Clinical Data ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val = scaler.transform(clinical_val[numeric_cols])
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat = clinical_val[categorical_cols]

    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat = torch.tensor(val_cat.values, dtype=torch.long)
    X_val_num = torch.tensor(val_num.values, dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train = clinical_train["PFI"].values.astype(bool)
    time_val = clinical_val["PFI.time"].values.astype(float)
    event_val = clinical_val["PFI"].values.astype(bool)

    # --- Build Unified Datasets ---
    unified_train_dataset = UnifiedDataset(
        clin_cat=X_train_cat,
        clin_num=X_train_num,
        times=time_train,
        events=event_train,
        wsi_dataset=ds_orig_train
    )
    unified_val_dataset = UnifiedDataset(
        clin_cat=X_val_cat,
        clin_num=X_val_num,
        times=time_val,
        events=event_val,
        wsi_dataset=ds_orig_val
    )
    batch_size = clin_param.get('batch_size', 64)
    unified_train_loader = DataLoader(unified_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=unified_collate_fn)
    unified_val_loader = DataLoader(unified_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)

    # --- Instantiate Branches & Fusion Modules ---
    cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]
    num_cont = len(numeric_cols)

    # Clinical branch parameters
    lr_clin = clin_param.get("lr", 1e-4)
    epochs_clin = clin_param.get("epochs", 20)
    dim_val = clin_param.get("dim", 64)
    depth_val = clin_param.get("depth", 6)
    heads_val = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val = clin_param.get("ff_dropout", 0.3)

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # WSI branch parameters
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_dropout = wsi_param.get('dropout', 0.0)
    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    # Fusion and final head
    fusion_module = ConcatenationFusion().to(device)
    final_surv_head = FinalSurvivalHead(input_dim=384).to(device)
    if cp_model == "crossattention":
        print("Running crossattention")
        fusion_module = CrossModalAttentionFusion(embed_dim=192, num_heads=2).to(device)
        final_surv_head = FinalSurvivalHead(input_dim=192).to(device)
    elif cp_model == "marginal":
        print("Running marginal")
        fusion_module = ConcatenationFusion().to(device)
        final_surv_head = FinalSurvivalHead(input_dim=384).to(device)
    

    unified_model = UnifiedSurvivalModel(
        clinical_model=clinical_model,
        wsi_model=wsi_model,
        fusion_module=fusion_module,
        final_head=final_surv_head
    ).to(device)

    # --- Define Optimizer & Loss ---
    optimizer = torch.optim.Adam(unified_model.parameters(), lr=lr_clin)
    loss_fn = CoxPHLoss()

    # ----- End-to-End Training Loop -----
    #unified_epochs = max(epochs_clin, wsi_param.get('epochs',50),50)
    unified_epochs = 100
    best_cindex = 0
    best_ibs_at_best_cindex = 0
    for epoch in range(unified_epochs):
        unified_model.train()
        total_loss = 0.0
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            if e_b.sum() == 0:
                print("Skipping batch with all censored cases")
                continue  # Skip this batch
            x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
            t_b, e_b = t_b.to(device), e_b.to(device)
            optimizer.zero_grad()
            risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            loss = loss_fn(risk, t_b, e_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Save model if it has the best C-index so far
        unified_model.eval()
        risks_list, times_list, events_list = [], [], []
        with torch.no_grad():
            for batch in unified_val_loader:
                x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
                risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
                risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
                risk = torch.clamp(risk, -1e6, 1e6)   
                risks_list.append(risk.cpu())
                times_list.append(t_b)
                events_list.append(e_b)
        final_risks = torch.cat(risks_list).cpu().numpy()
        final_times = torch.cat(times_list).cpu().numpy()
        final_events = torch.cat(events_list).cpu().numpy()

        cindex = concordance_index_censored(
            event_indicator=final_events.astype(bool),
            event_time=final_times,
            estimate=final_risks
        )[0]

        if cindex>best_cindex:
            if 'best_model_path' in locals() and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_cindex=cindex
            su1=uuid.uuid4().hex[:6]
            su2=uuid.uuid4().hex[:6]
            best_model_path=os.path.join(save_dir,f"best_model_epoch_{epoch}_{su1}_{su2}.pth")
            torch.save(unified_model.state_dict(),best_model_path)


        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(unified_train_loader)
            print(f"[Epoch {epoch+1}/{unified_epochs}] Loss: {avg_loss:.4f} | Validation C-Index: {cindex:.4f} (Best: {best_cindex:.4f})")

    # Compute IBS only for the best model
    unified_model.load_state_dict(torch.load(best_model_path))

    all_train_risks, all_train_times, all_train_events = [], [], []

    with torch.no_grad():
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch

            x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)

            risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   

            all_train_risks.append(risk.cpu().numpy())
            all_train_times.append(t_b.cpu().numpy())
            all_train_events.append(e_b.cpu().numpy())

    all_train_risks = np.concatenate(all_train_risks)
    all_train_times = np.concatenate(all_train_times)
    all_train_events = np.concatenate(all_train_events)

    # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
    from lifelines import CoxPHFitter
    train_df = pd.DataFrame({
        'duration': all_train_times,
        'event': all_train_events,
        'risk': all_train_risks
    })
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='duration', event_col='event')
    baseline_hazard_df = cph.baseline_cumulative_hazard_

    def baseline_cumulative_hazard(t):
        if t in baseline_hazard_df.index:
            return baseline_hazard_df.loc[t].values[0]
        else:
            return baseline_hazard_df.values[-1, 0]


    unified_model.eval()
    risks_list, times_list, events_list = [], [], []
    with torch.no_grad():
        for batch in unified_val_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
            risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            risk = torch.nan_to_num(risk, nan=0.0, posinf=1e6, neginf=-1e6)
            risk = torch.clamp(risk, -1e6, 1e6)   
            risks_list.append(risk.cpu())
            times_list.append(t_b)
            events_list.append(e_b)
    final_risks = torch.cat(risks_list).cpu().numpy()
    final_times = torch.cat(times_list).cpu().numpy()
    final_events = torch.cat(events_list).cpu().numpy()

    # Step 1: Prepare data
    risk_scores = final_risks  # Your risk scores
    times = final_times  # Event times
    events = final_events  # Event indicators (1 = event, 0 = censored)

    risk_scores = np.clip(risk_scores, -4, 4)  # First: control exp(risk)  
    # Step 2: Estimate baseline survival function
    baseline_surv = cph.baseline_survival_
    time_points = baseline_surv.index.values
    baseline_survival_at_times = baseline_surv['baseline survival'].values
    # Step 3: Compute time-dependent survival probabilities
    time_dependent_survival = np.array([
        baseline_survival_at_times ** np.exp(score) for score in risk_scores
    ])

    # Step 4: Create a DataFrame of survival probabilities
    # Each row corresponds to a patient, and each column corresponds to a time point
    survival_df = pd.DataFrame(time_dependent_survival, columns=time_points)

    #survival_df = survival_df.clip(lower=0.0, upper=1.0)
    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  
    survival_df = survival_df.apply(
        lambda row: pd.Series(np.minimum.accumulate(row.values), index=row.index),
        axis=1
    )

    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  

    # Step 5: Remove the last quartile of time points
    time_grid = time_points[time_points <= np.quantile(times[events == 1], 0.75)]

    survival_df.columns = survival_df.columns.astype(np.float32)
    time_grid = time_grid.astype(np.float32)

    # Step 6: Compute IBS using EvalSurv
    # Transpose survival_df to match EvalSurv's expected format
    ev = EvalSurv(survival_df.T, times, events, censor_surv='km')
    ibs = min(ev.integrated_brier_score(time_grid), 0.25)  # Compute IBS
    best_ibs = ibs
    print(f"[Unified Fusion] Best Validation C-Index: {best_cindex:.4f}, Corresponding IBS: {best_ibs:.4f}")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    return best_cindex, best_ibs

'''
# ----- Unified End-to-End VAE Fusion Training Function -----
def train_evaluate_end_to_end_fusion_vae(
    clinical_train,
    clinical_val,
    ds_orig_train,  # Subset of WSI dataset; each sample returns (regions, time, event)
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    device='cuda:1'
):
    """
    End-to-end training of the unified multimodal survival model with VAE-based joint fusion.
    Processes clinical and WSI inputs, aligns them via a unified dataset,
    instantiates the clinical branch, WSI branch, VAE fusion module, and final survival head,
    and trains the network using a combined loss: CoxPHLoss, reconstruction loss, and KL divergence.
    Returns the best validation concordance index.
    """
    # --- Process Clinical Data ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val = scaler.transform(clinical_val[numeric_cols])
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat = clinical_val[categorical_cols]

    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat = torch.tensor(val_cat.values, dtype=torch.long)
    X_val_num = torch.tensor(val_num.values, dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train = clinical_train["PFI"].values.astype(bool)
    time_val = clinical_val["PFI.time"].values.astype(float)
    event_val = clinical_val["PFI"].values.astype(bool)

    # --- Build Unified Datasets ---
    unified_train_dataset = UnifiedDataset(
        clin_cat=X_train_cat,
        clin_num=X_train_num,
        times=time_train,
        events=event_train,
        wsi_dataset=ds_orig_train
    )
    unified_val_dataset = UnifiedDataset(
        clin_cat=X_val_cat,
        clin_num=X_val_num,
        times=time_val,
        events=event_val,
        wsi_dataset=ds_orig_val
    )
    batch_size = clin_param.get('batch_size', 64)
    unified_train_loader = DataLoader(unified_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=unified_collate_fn)
    unified_val_loader = DataLoader(unified_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)

    # --- Instantiate Branches & Fusion Modules ---
    cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]
    num_cont = len(numeric_cols)

    # Clinical branch parameters
    lr_clin = clin_param.get("lr", 1e-4)
    epochs_clin = clin_param.get("epochs", 20)
    dim_val = clin_param.get("dim", 64)
    depth_val = clin_param.get("depth", 6)
    heads_val = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val = clin_param.get("ff_dropout", 0.3)

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # WSI branch parameters
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_dropout = wsi_param.get('dropout', 0.0)
    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    # VAE fusion parameters
    vae_latent_dim = wsi_param.get('vae_latent_dim', 192)  # set latent dimension for the VAE
    vae_hidden_dim = wsi_param.get('vae_hidden_dim', 256)
    vae_fusion = VAEFusion(input_dim=384, latent_dim=vae_latent_dim, hidden_dim=vae_hidden_dim).to(device)

    # Final survival head maps the latent space to risk score
    final_surv_head = FinalSurvivalHead(input_dim=vae_latent_dim).to(device)

    unified_model = UnifiedSurvivalModelVAE(
        clinical_model=clinical_model,
        wsi_model=wsi_model,
        vae_fusion=vae_fusion,
        final_head=final_surv_head
    ).to(device)

    # --- Define Optimizer & Loss ---
    optimizer = Adam(unified_model.parameters(), lr=lr_clin)
    loss_fn_surv = CoxPHLoss()
    mse_loss = nn.MSELoss(reduction='mean')
    
    # Hyperparameters for VAE loss components
    lambda_recon = wsi_param.get('lambda_recon', 1.0)
    beta = wsi_param.get('beta', 0.1)
    
    # ----- End-to-End Training Loop -----
    unified_epochs = max(epochs_clin, wsi_param.get('epochs',50), 80)
    best_cindex = 0
    best_ibs_at_best_cindex = 0
    for epoch in range(unified_epochs):
        unified_model.train()
        total_loss = 0.0
        total_recon_loss = 0.0  # Track reconstruction loss
        total_kl_loss = 0.0  # Track KL divergence loss
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
            t_b, e_b = t_b.to(device), e_b.to(device)
            
            optimizer.zero_grad()
            risk, x_recon, mu, logvar, cat_emb = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            # Compute survival loss
            loss_surv = loss_fn_surv(risk, t_b, e_b)
            # Reconstruction loss between reconstructed and original concatenated embeddings
            loss_recon = mse_loss(x_recon, cat_emb)
            # KL divergence loss: average over batch
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            loss_total = loss_surv + lambda_recon * loss_recon + beta * kl_loss
            loss_total.backward()
            optimizer.step()
            total_loss += loss_total.item()
            total_recon_loss += loss_recon.item()
            total_kl_loss += kl_loss.item()


            # --- Compute baseline cumulative hazard H_0(t) after each epoch --- #
            all_train_risks, all_train_times, all_train_events = [], [], []

            with torch.no_grad():
                for batch in unified_train_loader:
                    x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                    x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)

                    #risk = unified_model(x_cat, x_num, x_wsi, mask_wsi)
                    risk, _, _, _, _ = unified_model(x_cat, x_num, x_wsi, mask_wsi)
                    #print(risk)
                    #assert 1==0
                    all_train_risks.append(risk.cpu().numpy())
                    all_train_times.append(t_b.cpu().numpy())
                    all_train_events.append(e_b.cpu().numpy())

            all_train_risks = np.concatenate(all_train_risks)
            all_train_times = np.concatenate(all_train_times)
            all_train_events = np.concatenate(all_train_events)

            # --- USING lifelines CoxPHFitter instead of manual baseline hazard --- #
            from lifelines import CoxPHFitter
            train_df = pd.DataFrame({
                'duration': all_train_times,
                'event': all_train_events,
                'risk': all_train_risks
            })
            cph = CoxPHFitter()
            cph.fit(train_df, duration_col='duration', event_col='event')
            baseline_hazard_df = cph.baseline_cumulative_hazard_

            def baseline_cumulative_hazard(t):
                if t in baseline_hazard_df.index:
                    return baseline_hazard_df.loc[t].values[0]
                else:
                    return baseline_hazard_df.values[-1, 0]





        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(unified_train_loader)
            avg_recon_loss = total_recon_loss / len(unified_train_loader)
            avg_kl_loss = total_kl_loss / len(unified_train_loader)
            # ----- Validation after each epoch -----
            unified_model.eval()
            risks_list, times_list, events_list = [], [], []
            with torch.no_grad():
                for batch in unified_val_loader:
                    x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                    x_cat, x_num, x_wsi, mask_wsi = x_cat.to(device), x_num.to(device), x_wsi.to(device), mask_wsi.to(device)
                    risk_val, _, _, _, _ = unified_model(x_cat, x_num, x_wsi, mask_wsi)
                    risks_list.append(risk_val.cpu())
                    times_list.append(t_b)
                    events_list.append(e_b)
            final_risks = torch.cat(risks_list).numpy()
            final_times = torch.cat(times_list).numpy()
            final_events = torch.cat(events_list).numpy()

            # Convert to numpy
            risks_val = final_risks
            times_val = final_times
            events_val = final_events.astype(bool)


            # --- Compute IPCW weights for censored patients --- #
            kmf = KaplanMeierFitter()
            kmf.fit(times_val, event_observed=events_val == 0)  # Only censored patients

            time_points = np.sort(np.unique(times_val))  # Use actual event times
            ipcw_weights = {}
            n = len(times_val)  # Sample size for truncation formula
            lower_bound = 5 / (np.sqrt(n) * np.log(n))  # Sample-size-based lower bound

            for t in time_points:
                G_t = kmf.survival_function_at_times(t).values[0]
                G_t = max(G_t, lower_bound)
                ipcw_weights[t] = 1 / np.clip(G_t, 1e-8, None)  # Avoid division by zero

            # --- Compute IBS directly --- #
            ibs_values = []

            for t in time_points:
                H_t = baseline_cumulative_hazard(t)  # Get H_0(t)
                #S_t = np.exp(-H_t * risks_val)  # Convert risk scores to survival probabilities
                S_t = np.exp(-H_t)  # DO NOT multiply again by risks_val
                #print(f"Epoch {epoch}: S_t min={S_t.min()}, max={S_t.max()}")
            

                # Apply IPCW only to censored patients
                censor_mask = (events_val == 0)  # Mask for censored patients
                brier_t = np.mean(
                    (S_t - (times_val > t)) ** 2 * np.where(censor_mask, ipcw_weights[t], 1)
                )
                ibs_values.append(brier_t)

            # Compute final IBS as the mean of all Brier Scores across event times
            #print(ibs_values)
            ibs = np.mean(ibs_values)

            cindex = concordance_index_censored(
                event_indicator=events_val,
                event_time=times_val,
                estimate=risks_val
            )[0]

            if cindex > best_cindex:
                best_cindex = cindex
                best_ibs_at_best_cindex = ibs





            current_cindex = concordance_index(final_times, -final_risks, final_events)
            best_cindex = max(best_cindex, current_cindex)
            print(f"[Unified VAE Fusion] Epoch {epoch+1}/{unified_epochs}, Loss: {avg_loss:.4f}, Current Val C-index: {cindex:.4f} | IBS: {ibs:.4f} (IBS at Best C-Index: {best_ibs_at_best_cindex:.4f})")
            print(f"[Unified VAE Fusion] Epoch {epoch+1}/{unified_epochs}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
    
    print(f"[Unified VAE Fusion] Best Validation C-Index: {best_cindex:.4f}")
    return best_cindex, best_ibs_at_best_cindex

    '''
    # ----- Unified End-to-End VAE Fusion Training Function -----
def train_evaluate_end_to_end_fusion_vae(
    clinical_train,
    clinical_val,
    ds_orig_train,  # Subset of WSI dataset; each sample returns (regions, time, event)
    ds_orig_val,
    clin_param,
    wsi_param,
    numeric_cols,
    categorical_cols,
    clinical_common,
    save_dir,
    device='cuda:1'
):
    """
    End-to-end training of the unified multimodal survival model with VAE-based joint fusion.
    Processes clinical and WSI inputs, aligns them via a unified dataset,
    instantiates the clinical branch, WSI branch, VAE fusion module, and final survival head,
    and trains the network using a combined loss: CoxPHLoss, reconstruction loss, and KL divergence.
    Returns the best validation concordance index.
    """
    # --- Process Clinical Data ---
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(clinical_train[numeric_cols])
    scaled_val = scaler.transform(clinical_val[numeric_cols])
    train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=clinical_train.index)
    val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=clinical_val.index)
    train_cat = clinical_train[categorical_cols]
    val_cat = clinical_val[categorical_cols]

    X_train_cat = torch.tensor(train_cat.values, dtype=torch.long)
    X_train_num = torch.tensor(train_num.values, dtype=torch.float32)
    X_val_cat = torch.tensor(val_cat.values, dtype=torch.long)
    X_val_num = torch.tensor(val_num.values, dtype=torch.float32)

    time_train = clinical_train["PFI.time"].values.astype(float)
    event_train = clinical_train["PFI"].values.astype(bool)
    time_val = clinical_val["PFI.time"].values.astype(float)
    event_val = clinical_val["PFI"].values.astype(bool)

    # --- Build Unified Datasets ---
    unified_train_dataset = UnifiedDataset(
        clin_cat=X_train_cat,
        clin_num=X_train_num,
        times=time_train,
        events=event_train,
        wsi_dataset=ds_orig_train
    )
    unified_val_dataset = UnifiedDataset(
        clin_cat=X_val_cat,
        clin_num=X_val_num,
        times=time_val,
        events=event_val,
        wsi_dataset=ds_orig_val
    )
    batch_size = clin_param.get('batch_size', 64)
    unified_train_loader = DataLoader(unified_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=unified_collate_fn)
    unified_val_loader = DataLoader(unified_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=unified_collate_fn)

    # --- Instantiate Branches & Fusion Modules ---
    cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]
    num_cont = len(numeric_cols)

    # Clinical branch parameters
    lr_clin = clin_param.get("lr", 1e-4)
    epochs_clin = clin_param.get("epochs", 20)
    dim_val = clin_param.get("dim", 64)
    depth_val = clin_param.get("depth", 6)
    heads_val = clin_param.get("heads", 8)
    attn_drop_val = clin_param.get("attn_dropout", 0.3)
    ff_drop_val = clin_param.get("ff_dropout", 0.3)

    clinical_model = SurvivalModel_clinical(
        categories=cat_cardinalities,
        num_continuous=num_cont,
        dim=dim_val,
        depth=depth_val,
        heads=heads_val,
        attn_dropout=attn_drop_val,
        ff_dropout=ff_drop_val
    ).to(device)

    # WSI branch parameters
    wsi_hidden_dim = wsi_param.get('hidden_dim', 128)
    wsi_dropout = wsi_param.get('dropout', 0.0)
    wsi_model = SurvivalModel(
        input_dim=192,
        hidden_dim=wsi_hidden_dim,
        dropout=wsi_dropout
    ).to(device)

    # VAE fusion parameters
    vae_latent_dim = wsi_param.get('vae_latent_dim', 192)  # set latent dimension for the VAE
    vae_hidden_dim = wsi_param.get('vae_hidden_dim', 256)
    vae_fusion = VAEFusion(input_dim=384, latent_dim=vae_latent_dim, hidden_dim=vae_hidden_dim).to(device)

    # Final survival head maps the latent space to risk score
    final_surv_head = FinalSurvivalHead(input_dim=vae_latent_dim).to(device)

    unified_model = UnifiedSurvivalModelVAE(
        clinical_model=clinical_model,
        wsi_model=wsi_model,
        vae_fusion=vae_fusion,
        final_head=final_surv_head
    ).to(device)

    # --- Define Optimizer & Loss ---
    optimizer = Adam(unified_model.parameters(), lr=lr_clin)
    loss_fn_surv = CoxPHLoss()
    mse_loss = nn.MSELoss(reduction='mean')
    
    # Hyperparameters for VAE loss components
    lambda_recon = wsi_param.get('lambda_recon', 1.0)
    beta = wsi_param.get('beta', 0.1)
    
    # ----- End-to-End Training Loop -----
    unified_epochs = max(epochs_clin, wsi_param.get('epochs',50), 80)
    best_cindex = 0
    #best_ibs_at_best_cindex = 0
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_vae_fusion_model.pth")

    for epoch in range(unified_epochs):
        unified_model.train()
        total_loss, total_recon_loss, total_kl_loss = 0.0, 0.0, 0.0

        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch

            if e_b.sum() == 0:
                print("Skipping batch with all censoredpythoin cases")
                continue  # Skip this batch
            x_cat, x_num, x_wsi, mask_wsi = (
                x_cat.to(device),
                x_num.to(device),
                x_wsi.to(device),
                mask_wsi.to(device)
            )
            t_b, e_b = t_b.to(device), e_b.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            risk, x_recon, mu, logvar, cat_emb = unified_model(x_cat, x_num, x_wsi, mask_wsi)

            # Cox loss for survival
            loss_surv = loss_fn_surv(risk, t_b, e_b)

            # MSE reconstruction loss
            loss_recon = mse_loss(x_recon, cat_emb)

            # KL divergence
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

            # Weighted sum of all losses
            loss_total = loss_surv + lambda_recon * loss_recon + beta * kl_loss
            loss_total.backward()
            optimizer.step()

            total_loss       += loss_total.item()
            total_recon_loss += loss_recon.item()
            total_kl_loss    += kl_loss.item()

        # ----- Validation: compute C-index each epoch -----
        unified_model.eval()
        with torch.no_grad():
            risks_list, times_list, events_list = [], [], []
            for batch in unified_val_loader:
                x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
                x_cat, x_num, x_wsi, mask_wsi = (
                    x_cat.to(device),
                    x_num.to(device),
                    x_wsi.to(device),
                    mask_wsi.to(device)
                )
                risk_val, _, _, _, _ = unified_model(x_cat, x_num, x_wsi, mask_wsi)
                risks_list.append(risk_val.cpu())
                times_list.append(t_b)
                events_list.append(e_b)

            final_risks = torch.cat(risks_list).numpy()
            final_times = torch.cat(times_list).numpy()
            final_events = torch.cat(events_list).numpy().astype(bool)

        cindex = concordance_index_censored(
            event_indicator=final_events,
            event_time=final_times,
            estimate=final_risks
        )[0]

        # If this is our best C-index, save the model
        if cindex>best_cindex:
            if 'best_model_path' in locals() and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_cindex=cindex
            su1=uuid.uuid4().hex[:6]
            su2=uuid.uuid4().hex[:6]
            best_model_path=os.path.join(save_dir,f"best_model_epoch_{epoch}_{su1}_{su2}.pth")
            torch.save(unified_model.state_dict(),best_model_path)

        # Logging
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(unified_train_loader)
            avg_recon_loss = total_recon_loss / len(unified_train_loader)
            avg_kl_loss = total_kl_loss / len(unified_train_loader)

            print(f"[Unified VAE Fusion] Epoch {epoch+1}/{unified_epochs} "
                  f"- Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, "
                  f"Val C-index: {cindex:.4f} (Best: {best_cindex:.4f})")

    # ----------------------------------------------------------------------
    #               Compute IBS ONLY for the best model
    # ----------------------------------------------------------------------
    print(f"[Unified VAE Fusion] Finished training. Best Val C-Index = {best_cindex:.4f}")
    # 1) Load the best model
    unified_model.load_state_dict(torch.load(best_model_path))
    unified_model.eval()

    # 2) Fit baseline hazard on the *training* set risk scores
    all_train_risks, all_train_times, all_train_events = [], [], []
    with torch.no_grad():
        for batch in unified_train_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = (
                x_cat.to(device),
                x_num.to(device),
                x_wsi.to(device),
                mask_wsi.to(device)
            )
            risk, _, _, _, _ = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            all_train_risks.append(risk.cpu().numpy())
            all_train_times.append(t_b.numpy())
            all_train_events.append(e_b.numpy())

    all_train_risks = np.concatenate(all_train_risks)
    all_train_times = np.concatenate(all_train_times)
    all_train_events = np.concatenate(all_train_events)

    train_df = pd.DataFrame({
        'duration': all_train_times,
        'event': all_train_events,
        'risk': all_train_risks
    })
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='duration', event_col='event')
    baseline_hazard_df = cph.baseline_cumulative_hazard_

    def baseline_cumulative_hazard(t):
        # For discrete times, you may want interpolation.
        # Here, to match your existing approach, we just check or fall back to last row:
        if t in baseline_hazard_df.index:
            return baseline_hazard_df.loc[t].values[0]
        else:
            return baseline_hazard_df.values[-1, 0]

    # 3) Compute IBS on the validation set with the best model
    val_risks, val_times, val_events = [], [], []
    with torch.no_grad():
        for batch in unified_val_loader:
            x_cat, x_num, x_wsi, mask_wsi, t_b, e_b = batch
            x_cat, x_num, x_wsi, mask_wsi = (
                x_cat.to(device),
                x_num.to(device),
                x_wsi.to(device),
                mask_wsi.to(device)
            )
            risk, _, _, _, _ = unified_model(x_cat, x_num, x_wsi, mask_wsi)
            val_risks.append(risk.cpu().numpy())
            val_times.append(t_b.numpy())
            val_events.append(e_b.numpy())

    val_risks = np.concatenate(val_risks)
    val_times = np.concatenate(val_times)
    val_events = np.concatenate(val_events).astype(bool)

      # Step 1: Prepare data
    risk_scores = val_risks  # Your risk scores
    times = val_times  # Event times
    events = val_events  # Event indicators (1 = event, 0 = censored)

    risk_scores = np.clip(risk_scores, -4, 4)  # First: control exp(risk)  
    # Step 2: Estimate baseline survival function
    baseline_surv = cph.baseline_survival_
    time_points = baseline_surv.index.values
    baseline_survival_at_times = baseline_surv['baseline survival'].values
    # Step 3: Compute time-dependent survival probabilities
    time_dependent_survival = np.array([
        baseline_survival_at_times ** np.exp(score) for score in risk_scores
    ])

    # Step 4: Create a DataFrame of survival probabilities
    # Each row corresponds to a patient, and each column corresponds to a time point
    survival_df = pd.DataFrame(time_dependent_survival, columns=time_points)

    #survival_df = survival_df.clip(lower=0.0, upper=1.0)
    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  
    survival_df = survival_df.apply(
        lambda row: pd.Series(np.minimum.accumulate(row.values), index=row.index),
        axis=1
    )

    survival_df = survival_df.clip(1e-10, 1.0)  # Second: enforce final bounds  

    # Step 5: Remove the last quartile of time points
    time_grid = time_points[time_points <= np.quantile(times[events == 1], 0.75)]

    survival_df.columns = survival_df.columns.astype(np.float32)
    time_grid = time_grid.astype(np.float32)

    # Step 6: Compute IBS using EvalSurv
    # Transpose survival_df to match EvalSurv's expected format
    ev = EvalSurv(survival_df.T, times, events, censor_surv='km')
    ibs = min(ev.integrated_brier_score(time_grid), 0.25)  # Compute IBS
    best_ibs = ibs

    print(f"[Unified VAE Fusion] Best model reloaded. Final Val C-index = {best_cindex:.4f}, IBS = {ibs:.4f}")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    return best_cindex, ibs

###############################################################################
# Section D. Merged Joint Experiment Function (Standard CV + Per-Fold Contrastive Refinement)
###############################################################################

def run_joint_experiment_standard_cv(clinical_data_yes_id, clinical_param_grid, embedding_dir, json_path, wsi_param_grid, model_type, cp_model, sr, filter_substring="DX1"):
    """
    Runs standard 10-fold CV on clinical data and WSI embeddings.
    For each fold:
      1. Trains a clinical survival model on the training clinical data and computes risk scores.
      2. Determines the threshold as the median of the predicted risk scores on the training set.
      3. Generates binary labels for all patients in that fold using the training-set risk threshold.
      4. Updates a global JSON file ("svs_patient_map_PFI_blca_split.json") with a new column (e.g., "fold1_final") containing these labels.
      5. Runs contrastive learning (via run_contrastive_script) and refines embeddings (via refine_embeddings) using that column.
         The refined embeddings are saved in a folder named after the fold label (e.g., "./fold1_final").
      6. Evaluates CoxPH models on both original and refined WSI embeddings for that fold by training a CoxPH model on the fold’s embeddings.
    Returns a dictionary with per-fold results.
    """
    import os, json, numpy as np, torch, pandas as pd, random, warnings, itertools
    from tqdm import tqdm
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import Dataset, DataLoader, Subset
    from pycox.models.loss import CoxPHLoss
    from lifelines.utils import concordance_index
    from sklearn.preprocessing import StandardScaler
    from sksurv.metrics import concordance_index_censored
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import torchtuples as tt
    from pycox.models import CoxPH


    warnings.filterwarnings("ignore")
    torch.cuda.empty_cache()


    ###############################################################################
    # Preprocess Clinical Data
    ###############################################################################
    print(f"[INFO] Original clinical dataframe size: {len(clinical_data_yes_id)} rows")
    all_pt_files = [f for f in os.listdir(embedding_dir) if f.endswith('.pt') and filter_substring in f]
    barcodes_with_wsi = set(f[:12] for f in all_pt_files)
    clinical_data_filtered = clinical_data_yes_id[clinical_data_yes_id["patient.bcr_patient_barcode"].isin(barcodes_with_wsi)].copy()
    if clinical_data_filtered.empty:
        print("[WARNING] No matching clinical rows found!")
    else:
        print(f"[INFO] Filtered clinical dataframe has {len(clinical_data_filtered)} rows.")
    clinical_data_filtered["patient_id"] = clinical_data_filtered["patient.bcr_patient_barcode"]
    clinical_data_filtered.drop(columns=["patient.bcr_patient_barcode"], inplace=True)
    
    ###############################################################################
    # Preprocess WSI Data
    ###############################################################################
    # Instantiate SurvivalDataset (assumed defined elsewhere)
    wsi_dataset = SurvivalDataset(embedding_dir, json_path, filter_substring)
    for sample in wsi_dataset.valid_samples:
        # Use the first 12 characters of the filename as patient_id
        sample["patient_id"] = os.path.basename(sample["emb_path"])[:12]
    
    ###############################################################################
    # Compute Common Patients
    ###############################################################################
    clinical_ids = set(clinical_data_filtered["patient_id"].tolist())
    wsi_ids = set(sample["patient_id"] for sample in wsi_dataset.valid_samples)
    common_ids = clinical_ids.intersection(wsi_ids)
    print(f"[INFO] Number of common patients: {len(common_ids)}")
    clinical_common = clinical_data_filtered[clinical_data_filtered["patient_id"].isin(common_ids)].copy()
    # Keep a copy for WSI evaluation (with patient_id)
    clinical_common_wsi = clinical_common.copy()
    # Drop patient_id for survival modeling later
    clinical_common.drop(columns=["patient_id"], inplace=True)

    print(clinical_common['patient.days_to_birth'])

    ###############################################################################
    # Process Clinical Data (Numeric & Categorical)
    ###############################################################################
    numeric_cols = [
        'patient.days_to_birth',
        'patient.metastatic_breast_carcinoma_her2_neu_chromosone_17_signal_ratio_value',
        'patient.her2_neu_breast_carcinoma_copy_analysis_input_total_number',
        'patient.fluorescence_in_situ_hybridization_diagnostic_procedure_chromosome_17_signal_result_range',
        'patient.her2_neu_and_centromere_17_copy_number_analysis_input_total_number_count',
        'patient.breast_carcinoma_immunohistochemistry_pos_cell_score',
        'patient.immunohistochemistry_positive_cell_score',
        'patient.breast_carcinoma_immunohistochemistry_progesterone_receptor_pos_finding_scale',
        'patient.number_of_lymphnodes_positive_by_he',
        'patient.number_of_lymphnodes_positive_by_ihc',
        'patient.her2_immunohistochemistry_level_result',
        'patient.breast_carcinoma_immunohistochemistry_er_pos_finding_scale',
        'patient.her2_neu_chromosone_17_signal_ratio_value',
        'patient.lymph_node_examined_count'
    ]
    categorical_cols = [
        'patient.other_dx',
        'patient.gender',
        'patient.race_list.race',
        'patient.icd_o_3_site',
        'patient.icd_o_3_histology',
        'patient.icd_10',
        'patient.ethnicity',
        'patient.histological_type',
        'patient.tissue_prospective_collection_indicator',
        'patient.tissue_retrospective_collection_indicator',
        'patient.stage_event.pathologic_stage',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_T',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_N',
        'patient.stage_event.tnm_categories.pathologic_categories.pathologic_M',
        'patient.first_nonlymph_node_metastasis_anatomic_sites.metastatic_site_at_diagnosis_4',
        'patient.metastatic_breast_carcinoma_immunohistochemistry_pr_pos_cell_score',
        'patient.metastatic_breast_carcinoma_immunohistochemistry_er_pos_cell_score',
        'patient.metastatic_breast_carcinoma_lab_proc_her2_neu_in_situ_hybridization_outcome_type',
        'patient.first_nonlymph_node_metastasis_anatomic_sites.metastatic_site_at_diagnosis_other',
        'patient.first_nonlymph_node_metastasis_anatomic_sites.metastatic_site_at_diagnosis_3',
        'patient.first_nonlymph_node_metastasis_anatomic_sites.metastatic_site_at_diagnosis_2',
        'patient.her2_and_centromere_17_positive_finding_other_measurement_scale_text',
        'patient.anatomic_neoplasm_subdivisions.anatomic_neoplasm_subdivision_5',
        'patient.breast_neoplasm_other_surgical_procedure_descriptive_text',
        'patient.axillary_lymph_node_stage_other_method_descriptive_text',
        'patient.breast_carcinoma_primary_surgical_procedure_name',
        'patient.anatomic_neoplasm_subdivisions.anatomic_neoplasm_subdivision_4',
        'patient.metastatic_breast_carcinoma_erbb2_immunohistochemistry_level_result',
        'patient.metastatic_breast_carcinoma_her2_erbb_pos_finding_cell_percent_category',
        'patient.metastatic_breast_carcinoma_progesterone_receptor_level_cell_percent_category',
        'patient.metastatic_breast_carcinoma_lab_proc_her2_neu_immunohistochemistry_receptor_status',
        'patient.metastatic_breast_carcinoma_progesterone_receptor_status',
        'patient.metastatic_breast_carcinoma_estrogen_receptor_level_cell_percent_category',
        'patient.metastatic_breast_carcinoma_estrogen_receptor_status',
        'patient.first_nonlymph_node_metastasis_anatomic_sites.metastatic_site_at_diagnosis',
        'patient.histological_type_other',
        'patient.pos_finding_her2_erbb2_other_measurement_scale_text',
        'patient.anatomic_neoplasm_subdivisions.anatomic_neoplasm_subdivision_3',
        'patient.breast_cancer_surgery_margin_status',
        'patient.positive_finding_estrogen_receptor_other_measurement_scale_text',
        'patient.pos_finding_progesterone_receptor_other_measurement_scale_text',
        'patient.her2_erbb_pos_finding_fluorescence_in_situ_hybridization_calculation_method_text',
        'patient.her2_erbb_method_calculation_method_text',
        'patient.anatomic_neoplasm_subdivisions.anatomic_neoplasm_subdivision_2',
        'patient.init_pathology_dx_method_other',
        'patient.surgical_procedure_purpose_other_text',
        'patient.pgr_detection_method_text',
        'patient.er_detection_method_text',
        'patient.stage_event.system_version',
        'patient.distant_metastasis_present_ind2',
        'patient.progesterone_receptor_level_cell_percent_category',
        'patient.er_level_cell_percentage_category',
        'patient.her2_erbb_pos_finding_cell_percent_category',
        'patient.lab_proc_her2_neu_immunohistochemistry_receptor_status',
        'patient.breast_carcinoma_estrogen_receptor_status',
        'patient.lab_procedure_her2_neu_in_situ_hybrid_outcome_type',
        'patient.initial_pathologic_diagnosis_method',
        'patient.margin_status',
        'patient.cytokeratin_immunohistochemistry_staining_method_micrometastasis_indicator',
        'patient.breast_carcinoma_progesterone_receptor_status',
        'patient.menopause_status',
        'patient.breast_carcinoma_surgical_procedure_name',
        'patient.axillary_lymph_node_stage_method_type',
        'patient.anatomic_neoplasm_subdivisions.anatomic_neoplasm_subdivision',
        'patient.primary_lymph_node_presentation_assessment'
    ]

    for col in numeric_cols:
        if col in clinical_common.columns:
            clinical_common[col] = pd.to_numeric(clinical_common[col], errors='coerce')
            #clinical_common[col].fillna(clinical_common[col].mean(), inplace=True)
            clinical_common[col].fillna(-1, inplace=True)
    for col in categorical_cols:
        if col in clinical_common.columns:
            clinical_common[col] = clinical_common[col].fillna("missing")
            clinical_common[col] = clinical_common[col].astype('category').cat.codes
    if 'PFI.time' in clinical_common.columns:
        clinical_common['PFI.time'].fillna(clinical_common['PFI.time'].mean(), inplace=True)
    if 'PFI' in clinical_common.columns:
        clinical_common['PFI'].fillna(0, inplace=True)
    time_all = clinical_common['PFI.time'].values.astype(float)
    event_all = clinical_common['PFI'].values.astype(bool)
    
    ###############################################################################
    # Generate 10-Fold Splits
    ###############################################################################





    # Create survival time bins (convert continuous time into categories)
    num_bins = 5  # Adjust based on dataset size
    clinical_common["PFI_time_bins"] = pd.qcut(clinical_common["PFI.time"], num_bins, labels=False)
    print(clinical_common["PFI_time_bins"].value_counts())
 


    # Combine event status and binned survival time into a single stratification label
    stratify_labels = clinical_common["PFI"].astype(str) + "_" + clinical_common["PFI_time_bins"].astype(str)

    # Now do stratified K-fold using this combined label
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=sr)
    fold_splits = list(skf.split(clinical_common, stratify_labels))


    for fold_num, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"Fold {fold_num}: Train Min={clinical_common.iloc[train_idx]['PFI.time'].min()}, Max={clinical_common.iloc[train_idx]['PFI.time'].max()}")
        print(f"Fold {fold_num}: Val Min={clinical_common.iloc[val_idx]['PFI.time'].min()}, Max={clinical_common.iloc[val_idx]['PFI.time'].max()}")
    t.sleep(4)
    
    ###############################################################################
    # Load Original WSI Embeddings (region-level)
    ###############################################################################
    missing_files = []
    # Change the lookup to load region-level embeddings.
    #orig_embeddings_dir = os.path.expanduser('/Data/Juan/local_embeddings/TCGA-BRCA-embeddings/')
    #orig_embeddings_dir = os.path.expanduser('/home/sorkwos/projects/rrg-senger-ab/multimodality/contrastive_learning/TCGA-HNSC-data/local_embeddings/TCGA-BRCA-embeddings/')
    orig_embeddings_dir = os.path.abspath('../local_embeddings/TCGA-BRCA-embeddings/')       
    def load_embedding_parallel(file_name):
        # Expect region-level embeddings saved as .pt (not flattened)
        embedding_file = file_name.replace('.svs', '.pt')
        embedding_path = os.path.join(orig_embeddings_dir, embedding_file)
        print(embedding_path)
        if not os.path.exists(embedding_path):
            print(f"[WARNING] File does NOT exist: {embedding_path}")  # This will show if paths are wrong
            missing_files.append(embedding_file)
            return None
        try:
            embedding_tensor = torch.load(embedding_path)
            # Return the regioncany -level embedding (shape: n x 192)
            #return embedding_tensor.numpy().astype(np.float32)
            return embedding_tensor.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to load {embedding_path}: {str(e)}")  # Show error message
            missing_files.append(embedding_file)
            return None
    def load_embeddings_in_parallel(wsi_df_local):
        embeddings_local = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(load_embedding_parallel, row['file_name']): idx 
                       for idx, row in wsi_df_local.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading original WSI embeddings"):
                embeddings_local.append(future.result())

        return embeddings_local
    wsi_df = pd.DataFrame([{"file_name": os.path.basename(s["emb_path"]).replace('_flatten.pt', '.svs'),
                             "patient.bcr_patient_barcode": s["patient_id"]}
                            for s in wsi_dataset.valid_samples])

    print("Len before dropna is ",len(wsi_df))
    wsi_df = wsi_df[wsi_df['file_name'].str.contains('DX1', case=False, na=False)].reset_index(drop=True)
    print("Len after dropna is ",len(wsi_df))
    

    wsi_df['embedding'] = load_embeddings_in_parallel(wsi_df)

    wsi_df = wsi_df[wsi_df['embedding'].notnull()].reset_index(drop=True)
    barcode_to_wsi = dict(zip(wsi_df['patient.bcr_patient_barcode'], wsi_df['embedding']))

    barcode_to_wsi = {s["patient_id"]: torch.load(s["emb_path"]).cpu().numpy().astype(np.float32)
                    for s in wsi_dataset.valid_samples}

    patient_barcodes_full = pd.Series(list(common_ids))
    
    ###############################################################################
    # Prepare containers for results
    ###############################################################################
    clinical_results = []
    clinical_results_ibs = []
    wsi_results_original = []
    wsi_results_original_ibs = []
    unified_results = []
    unified_results_ibs = []
    
    ###############################################################################
    # Global JSON for splits: updated with new binary label columns per fold.
    ###############################################################################
    split_json_path = "svs_patient_map_PFI_blca_split.json"
    if os.path.exists(split_json_path):
        global_split_df = pd.read_json(split_json_path)
    else:
        global_split_df = clinical_common_wsi.copy()

    ###############################################################################
    # Loop Over Each Fold
    ###############################################################################
    for fold_num, (train_idx, val_idx) in enumerate(fold_splits, start=1):
        print(f"\n====== Fold {fold_num} ======")
        clinical_train = clinical_common.iloc[train_idx].copy()
        clinical_val = clinical_common.iloc[val_idx].copy()
        
        # --- Train Clinical Survival Model on Training Data and Evaluate on Validation Data ---
        clin_param = {k: clinical_param_grid[k][0] for k in clinical_param_grid}

 
        def run_clinical_fold(train_df, val_df, param_dict):

            # Scale numeric features using StandardScaler
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_df[numeric_cols])
            scaled_val = scaler.transform(val_df[numeric_cols])
            train_num = pd.DataFrame(scaled_train, columns=numeric_cols, index=train_df.index)
            val_num = pd.DataFrame(scaled_val, columns=numeric_cols, index=val_df.index)
            train_cat = train_df[categorical_cols]
            val_cat = val_df[categorical_cols]


            #def prepare_data(df, cat_cols, num_cols):
            #    return df[cat_cols], df[num_cols]
            #X_train_cat, X_train_num = prepare_data(pd.concat([train_num, train_cat], axis=1), categorical_cols, numeric_cols)
            #X_val_cat, X_val_num = prepare_data(pd.concat([val_num, val_cat], axis=1), categorical_cols, numeric_cols)
            
            # Prepare data without unnecessary concatenation
            def prepare_data(cat_df, num_df, cat_cols, num_cols):
                return cat_df[cat_cols], num_df[num_cols]

            X_train_cat, X_train_num = prepare_data(train_cat, train_num, categorical_cols, numeric_cols)
            X_val_cat, X_val_num = prepare_data(val_cat, val_num, categorical_cols, numeric_cols)
    
            # Create tensors for features
            X_train_cat_tensor = torch.tensor(X_train_cat.values, dtype=torch.long)
            X_train_num_tensor = torch.tensor(X_train_num.values, dtype=torch.float32)
            X_val_cat_tensor = torch.tensor(X_val_cat.values, dtype=torch.long)
            X_val_num_tensor = torch.tensor(X_val_num.values, dtype=torch.float32)
            
            # Extract survival times and events
            time_train = train_df["PFI.time"].values.astype(float)
            event_train = train_df["PFI"].values.astype(bool)
            time_val = val_df["PFI.time"].values.astype(float)
            event_val = val_df["PFI"].values.astype(bool)
            
            # Build datasets (note: for evaluation we will use non-shuffled loaders)
            train_dataset = torch.utils.data.TensorDataset(X_train_cat_tensor, X_train_num_tensor,
                                                            torch.FloatTensor(time_train),
                                                            torch.FloatTensor(event_train))
            val_dataset = torch.utils.data.TensorDataset(X_val_cat_tensor, X_val_num_tensor,
                                                        torch.FloatTensor(time_val),
                                                        torch.FloatTensor(event_val))
            train_loader = DataLoader(train_dataset, batch_size=param_dict.get('batch_size', 64), shuffle=True)
            train_loader_eval = DataLoader(train_dataset, batch_size=param_dict.get('batch_size', 64), shuffle=False)
            val_loader_eval = DataLoader(val_dataset, batch_size=param_dict.get('batch_size', 64), shuffle=False)
            
            # Compute cardinalities and number of continuous features for the clinical model
            cat_cardinalities = [int(clinical_common[col].max() + 1) for col in categorical_cols if col in clinical_common.columns]

            num_cont_features = len(numeric_cols)


            # Optionally, compute a C-index via your helper (this line is kept as in your original code)
            cindex,clinical_ibs = train_evaluate_fold_clinical(param_dict, train_loader_eval, val_loader_eval, 
                                                cat_cardinalities, num_cont_features, "saved_models", device='cuda')
            
            return cindex,clinical_ibs
        
        # Run training on the fold (returns C-index, risk scores on train and val, the trained model, and the scaler)
        clin_cindex, clin_ibs = run_clinical_fold(clinical_train, clinical_val, clin_param)
        print(f"Clinical C-Index for fold {fold_num}: {clin_cindex:.4f}")
        clinical_results.append(clin_cindex)
        clinical_results_ibs.append(clin_ibs)
        


        
        # Update the global split JSON with the new binary labels.
        mapping_df = clinical_common_wsi[["patient_id", "PFI"]].copy()

        
        # Create a mapping from patient_id to file_name from wsi_dataset
        patient_to_filename = {s["patient_id"]: os.path.basename(s["emb_path"]).replace('.pt', '.svs') for s in wsi_dataset.valid_samples}
        mapping_df["file_name"] = mapping_df["patient_id"].map(patient_to_filename)
        
        global_split_df = mapping_df.copy()
    


        # --- Map patient IDs to indices ---
        wsi_patient_ids = [sample["patient_id"] for sample in wsi_dataset.valid_samples]
        patient_to_index = {pid: idx for idx, pid in enumerate(wsi_patient_ids)}

        train_patient_ids = clinical_common_wsi.iloc[train_idx]["patient_id"].tolist()
        val_patient_ids = clinical_common_wsi.iloc[val_idx]["patient_id"].tolist()

        train_indices = [patient_to_index[pid] for pid in train_patient_ids if pid in patient_to_index]
        val_indices = [patient_to_index[pid] for pid in val_patient_ids if pid in patient_to_index]


        '''
        for pid in train_patient_ids[:5]:  # Print for the first 5 patients in the training fold
            if pid in barcode_to_wsi:
                # Get the correct row by indexing with patient_id from clinical_common_wsi
                clinical_emb = clinical_common_wsi[clinical_common_wsi["patient_id"] == pid].drop(columns=["patient_id"]).values
                wsi_emb = barcode_to_wsi[pid]
                
                print(f"Patient ID: {pid}")
                print(f"Clinical Embedding Shape: {clinical_emb.shape}")
                print(f"WSI Embedding Shape: {wsi_emb.shape}")
                print(f"Clinical Embedding: {clinical_emb}")
                print(f"WSI Embedding: {wsi_emb}")
                assert 1 == 0  # Force stop for debugging
        '''




        # --- Create Train & Validation Datasets ---
        ds_orig_train = Subset(wsi_dataset, train_indices)

        ds_orig_val = Subset(wsi_dataset, val_indices)


        wsi_param = {k: wsi_param_grid[k][0] for k in wsi_param_grid}

        generator = torch.Generator()
        generator.manual_seed(42)
        # --- DataLoaders ---
        train_loader_orig = DataLoader(
            ds_orig_train,
            batch_size=wsi_param.get('batch_size', 64),
            collate_fn=collate_fn,
            shuffle=True,
            generator=generator
        )

        val_loader_orig = DataLoader(
            ds_orig_val,
            batch_size=wsi_param.get('batch_size', 64),
            collate_fn=collate_fn,
            shuffle=False
        )

        # --- Train & Evaluate ---



        #model_type = "intermediate"
        #cp_model = "marginal"

        if model_type == "intermediate":       
            unified_cindex, unified_ibs = train_evaluate_end_to_end_fusion(
                cp_model,
                clinical_train,
                clinical_val,
                ds_orig_train,  # These are unified WSI datasets aligned with clinical data (subsets)
                ds_orig_val,
                clin_param,
                wsi_param,
                numeric_cols,
                categorical_cols,
                clinical_common,
                "saved_models",
                device
            )
        elif model_type == "vae":
            print("training VAE")
            unified_cindex, unified_ibs = train_evaluate_end_to_end_fusion_vae(
                clinical_train,
                clinical_val,
                ds_orig_train,  # These are unified WSI datasets aligned with clinical data (subsets)
                ds_orig_val,
                clin_param,
                wsi_param,
                numeric_cols,
                categorical_cols,
                clinical_common,
                "saved_models",
                device
            )
        elif model_type == "late":
        
            unified_cindex, unified_ibs = train_evaluate_end_to_end_fusion_late(
                cp_model,
                clinical_train,
                clinical_val,
                ds_orig_train,
                ds_orig_val,
                clin_param,
                wsi_param,
                numeric_cols,
                categorical_cols,
                clinical_common,
                "saved_models",
                device
            )

        unified_results.append(unified_cindex)
        unified_results_ibs.append(unified_ibs)
        orig_cindex, wsi_ibs= train_evaluate_wsi(wsi_param, train_loader_orig, val_loader_orig, "saved_models")
        print(f"Fold {fold_num}: Original WSI C-Index = {orig_cindex:.4f}")
        print(wsi_ibs)
        wsi_results_original.append(orig_cindex)
        wsi_results_original_ibs.append(wsi_ibs)
    
        # Final results
        results = {
            "clinical_results": clinical_results,
            "wsi_results_original": wsi_results_original,
            "unified_results": unified_results,
            "clinical_results_ibs": clinical_results_ibs,
            "wsi_results_original_ibs": wsi_results_original_ibs,
            "unified_results_ibs": unified_results_ibs
        }
    return results


###############################################################################
# Section F. Main Function Call
###############################################################################

import numpy as np
from scipy import stats

if __name__ == "__main__":
    # Define hyperparameter grids for clinical and WSI models.
    my_clin_param_grid = {
        "lr": [1e-4],
        "epochs": [100],
        "dim": [192],
        "depth": [6],
        "heads": [8],
        "attn_dropout": [0.3],
        "ff_dropout": [0.3],
        "batch_size": [32]
    }
    my_wsi_param_grid = {
        "lr": [1e-3],
        "dropout": [0.0],
        "hidden_dim": [128],
        "weight_decay": [1e-5],
        "epochs": [100],
        "batch_size": [32],
        "grad_clip": [1.0]
    }

    #embedding_dir = "/Data/Juan/local_embeddings/TCGA-BRCA-embeddings/"
    #embedding_dir = "/home/sorkwos/projects/rrg-senger-ab/multimodality/contrastive_learning/TCGA-HNSC-data/local_embeddings/TCGA-BRCA-embeddings/"
    embedding_dir = '../local_embeddings/TCGA-BRCA-embeddings/'     
    json_path = "svs_patient_map_PFI_brca.json"

    cancer_types = ["BRCA", "BLCA", "HNSC", "UCEC", "LUAD"]
    script_dataset = None
    for cancer in cancer_types:
        if cancer in embedding_dir:
            script_dataset = cancer
            break  # Stop at first match
    print(f"Detected dataset: {script_dataset}") if script_dataset else print("No known cancer type found in path")

# Initialize storage for RAW FOLD-LEVEL RESULTS
raw_clinical = []
raw_wsi = []
raw_unified = []
raw_clinical_ibs = []
raw_wsi_ibs = []
raw_unified_ibs = []

# Initialize storage for run-level statistics
all_clin_means, all_clin_stds = [], []
all_wsi_means, all_wsi_stds = [], []
all_uni_means, all_uni_stds = [], []
all_clin_ibs_means, all_clin_ibs_stds = [], []
all_wsi_ibs_means, all_wsi_ibs_stds = [], []
all_uni_ibs_means, all_uni_ibs_stds = [], []

num_runs = 100
model_type = "intermediate"
cp_model = "marginal"
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{script_dataset}_{model_type}_{cp_model}_{date_str}.txt" if model_type != "vae" else f"{script_dataset}_{model_type}_{date_str}.txt"

for run in range(num_runs):
    print(f"=== Running 10-Fold CV: Iteration {run + 1}/{num_runs} ===")

    results = run_joint_experiment_standard_cv(
        clinical_data_yes_id=clinical_data_yes_id,
        clinical_param_grid=my_clin_param_grid,
        embedding_dir=embedding_dir,
        json_path=json_path,
        wsi_param_grid=my_wsi_param_grid,
        model_type = model_type,
        cp_model = cp_model,
        sr = run,
        filter_substring="DX1",
    )

    # Store RAW FOLD-LEVEL DATA
    raw_clinical.append(results["clinical_results"])
    raw_wsi.append(results["wsi_results_original"])
    raw_unified.append(results["unified_results"])
    raw_clinical_ibs.append(results["clinical_results_ibs"])
    raw_wsi_ibs.append(results["wsi_results_original_ibs"])
    raw_unified_ibs.append(results["unified_results_ibs"])

    # Calculate and store run-level statistics
    def store_run_stats(source, means, stds):
        means.append(np.mean(source))
        stds.append(np.std(source, ddof=1))
    
    store_run_stats(results["clinical_results"], all_clin_means, all_clin_stds)
    store_run_stats(results["wsi_results_original"], all_wsi_means, all_wsi_stds)
    store_run_stats(results["unified_results"], all_uni_means, all_uni_stds)
    store_run_stats(results["clinical_results_ibs"], all_clin_ibs_means, all_clin_ibs_stds)
    store_run_stats(results["wsi_results_original_ibs"], all_wsi_ibs_means, all_wsi_ibs_stds)
    store_run_stats(results["unified_results_ibs"], all_uni_ibs_means, all_uni_ibs_stds)

# ==================================================================
# SAVE EVERYTHING TO SINGLE FILE
# ==================================================================
with open(output_filename, 'w') as f:
    # Write raw data section
    f.write("=== RAW FOLD-LEVEL RESULTS ===\n")
    
    def write_raw(f, name, data):
        f.write(f"\n{name}:\n")
        for run_idx, run_data in enumerate(data):
            f.write(f"Run {run_idx+1}: " + ", ".join([f"{x:.6f}" for x in run_data]) + "\n")
    
    write_raw(f, "Clinical C-Index", raw_clinical)
    write_raw(f, "WSI C-Index", raw_wsi)
    write_raw(f, "Unified C-Index", raw_unified)
    write_raw(f, "Clinical IBS", raw_clinical_ibs)
    write_raw(f, "WSI IBS", raw_wsi_ibs)
    write_raw(f, "Unified IBS", raw_unified_ibs)
    
    # Write statistics section
    f.write("\n\n=== STATISTICAL ANALYSIS ===\n")
    f.write(f"Dataset: {script_dataset}\n")
    f.write(f"Model: {model_type} ({cp_model})\n")
    f.write(f"Total Runs: {num_runs}\n\n")
    
    def write_stats(f, name, stats):
        f.write(f"{name}:\n")
        f.write(f"  Mean: {stats['mean']:.4f} ± {stats['total_sd']:.4f}\n")
        f.write(f"  95% CI: ±{stats['ci']:.4f}\n")
        f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n\n")
    
    # Compute statistics
    def compute_stats(means, stds):
        grand_mean = np.mean(means)
        between_sd = np.std(means, ddof=1)
        within_sd = np.mean(stds)
        total_sd = np.sqrt(between_sd**2 + (within_sd**2)/10)
        ci = 1.96 * total_sd / np.sqrt(len(means))
        return {
            'mean': grand_mean,
            'total_sd': total_sd,
            'ci': ci,
            'min': np.min(means),
            'max': np.max(means)
        }
    
    write_stats(f, "Clinical C-Index", compute_stats(all_clin_means, all_clin_stds))
    write_stats(f, "WSI C-Index", compute_stats(all_wsi_means, all_wsi_stds))
    write_stats(f, "Unified C-Index", compute_stats(all_uni_means, all_uni_stds))
    write_stats(f, "Clinical IBS", compute_stats(all_clin_ibs_means, all_clin_ibs_stds))
    write_stats(f, "WSI IBS", compute_stats(all_wsi_ibs_means, all_wsi_ibs_stds))
    write_stats(f, "Unified IBS", compute_stats(all_uni_ibs_means, all_uni_ibs_stds))

print(f"\n=== ALL RESULTS SAVED TO: {output_filename} ===")