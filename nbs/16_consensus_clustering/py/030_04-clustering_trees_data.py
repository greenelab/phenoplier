# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It prepares the data to create a clustering tree visualization (using the R package `clustree`).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import numpy as np
import pandas as pd

from utils import generate_result_set_name
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## PCA

# %% tags=[]
INPUT_SUBSET = "pca"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data_pca = pd.read_pickle(input_filepath).iloc[:, :5]

# %% tags=[]
data_pca.shape

# %% tags=[]
data_pca.head()

# %% [markdown] tags=[]
# ## UMAP

# %% tags=[]
INPUT_SUBSET = "umap"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 5,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data_umap = pd.read_pickle(input_filepath)

# %% tags=[]
data_umap.shape

# %% tags=[]
data_umap.head()

# %% [markdown] tags=[]
# # Load selected best partitions

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %% [markdown] tags=[]
# # Prepare data for clustrees

# %% tags=[]
clustrees_df = pd.concat((data_pca, data_umap), join="inner", axis=1)

# %% tags=[]
display(clustrees_df.shape)
assert clustrees_df.shape == (data_pca.shape[0], data_pca.shape[1] + data_umap.shape[1])

# %% tags=[]
clustrees_df.head()

# %% [markdown] tags=[]
# ## Add partitions

# %% tags=[]
_tmp = np.unique(
    [best_partitions.loc[k, "partition"].shape for k in best_partitions.index]
)
display(_tmp)
assert _tmp.shape[0] == 1
assert _tmp[0] == data_umap.shape[0] == data_pca.shape[0]

# %% tags=[]
assert not best_partitions.isna().any().any()

# %% tags=[]
# df = df.assign(**{f'k{k}': partitions.loc[k, 'partition'] for k in selected_k_values})
clustrees_df = clustrees_df.assign(
    **{
        f"k{k}": best_partitions.loc[k, "partition"]
        for k in best_partitions.index
        if best_partitions.loc[k, "selected"]
    }
)

# %% tags=[]
clustrees_df.index.rename("trait", inplace=True)

# %% tags=[]
clustrees_df.shape

# %% tags=[]
clustrees_df.head()

# %% tags=[]
# make sure partitions were assigned correctly
assert (
    np.unique(
        [
            clustrees_df[f"{k}"].value_counts().sum()
            for k in clustrees_df.columns[
                clustrees_df.columns.str.contains("^k[0-9]+$", regex=True)
            ]
        ]
    )[0]
    == data_pca.shape[0]
)

# %% [markdown] tags=[]
# # Assign labels

# %% tags=[]
trait_labels = pd.Series({t: None for t in clustrees_df.index})

# %% tags=[]
trait_labels.head()

# %% tags=[]
trait_labels.loc["3143_raw-Ankle_spacing_width"] = "Anthropometry"

trait_labels.loc[
    [
        "20153_raw-Forced_expiratory_volume_in_1second_FEV1_predicted",
        "20150_raw-Forced_expiratory_volume_in_1second_FEV1_Best_measure",
        "20151_raw-Forced_vital_capacity_FVC_Best_measure",
        "3062_raw-Forced_vital_capacity_FVC",
        "3063_raw-Forced_expiratory_volume_in_1second_FEV1",
    ]
] = "Spirometry"


trait_labels.loc[
    [
        "30080_raw-Platelet_count",
        "30090_raw-Platelet_crit",
        "30100_raw-Mean_platelet_thrombocyte_volume",
        "30110_raw-Platelet_distribution_width",
        "platelet count",
    ]
] = "Platelet"

trait_labels.loc[
    [
        "23106_raw-Impedance_of_whole_body",
        "23107_raw-Impedance_of_leg_right",
        "23108_raw-Impedance_of_leg_left",
        "23109_raw-Impedance_of_arm_right",
        "23110_raw-Impedance_of_arm_left",
    ]
] = "Impedance"

trait_labels.loc[
    [
        "40001_C349-Underlying_primary_cause_of_death_ICD10_C349_Bronchus_or_lung_unspecified",
        "C3_RESPIRATORY_INTRATHORACIC-Malignant_neoplasm_of_respiratory_system_and_intrathoracic_organs",
        "C_BRONCHUS_LUNG-Malignant_neoplasm_of_bronchus_and_lung",
        "C_RESPIRATORY_INTRATHORACIC",
        "LUNG_CANCER_MESOT-Lung_cancer_and_mesothelioma",
        "lung carcinoma",
    ]
] = "Lung cancer"

# From https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=100014
trait_labels.loc[
    [
        "5086_raw-Cylindrical_power_left",
        "5087_raw-Cylindrical_power_right",
        "5116_raw-3mm_cylindrical_power_right",
        "5117_raw-6mm_cylindrical_power_right",
        "5118_raw-6mm_cylindrical_power_left",
        "5119_raw-3mm_cylindrical_power_left",
    ]
] = "Refractometry"

trait_labels.loc[
    [
        "5096_raw-3mm_weak_meridian_left",
        "5097_raw-6mm_weak_meridian_left",
        "5098_raw-6mm_weak_meridian_right",
        "5099_raw-3mm_weak_meridian_right",
        "5132_raw-3mm_strong_meridian_right",
        "5133_raw-6mm_strong_meridian_right",
        "5134_raw-6mm_strong_meridian_left",
        "5135_raw-3mm_strong_meridian_left",
    ]
] = "Keratometry"

trait_labels.loc[
    [
        "3144_raw-Heel_Broadband_ultrasound_attenuation_direct_entry",
        "3147_raw-Heel_quantitative_ultrasound_index_QUI_direct_entry",
        "3148_raw-Heel_bone_mineral_density_BMD",
        "4101_raw-Heel_broadband_ultrasound_attenuation_left",
        "4104_raw-Heel_quantitative_ultrasound_index_QUI_direct_entry_left",
        "4105_raw-Heel_bone_mineral_density_BMD_left",
        "4106_raw-Heel_bone_mineral_density_BMD_Tscore_automated_left",
        "4120_raw-Heel_broadband_ultrasound_attenuation_right",
        "4123_raw-Heel_quantitative_ultrasound_index_QUI_direct_entry_right",
        "4124_raw-Heel_bone_mineral_density_BMD_right",
        "4125_raw-Heel_bone_mineral_density_BMD_Tscore_automated_right",
        "78_raw-Heel_bone_mineral_density_BMD_Tscore_automated",
    ]
] = "Heel bone"


trait_labels.loc[
    [
        "22617_3319-Job_SOC_coding_Protective_service_associate_professionals_nec",
        "5983_raw-ECG_heart_rate",
        "5984_raw-ECG_load",
        "5986_raw-ECG_phase_time",
        "5992-ECG_phase_duration",
        "5993-ECG_number_of_stages_in_a_phase",
        "6020_1-Completion_status_of_test_Fully_completed",
        "6020_31-Completion_status_of_test_Participant_wanted_to_stop_early",
        "6020_33-Completion_status_of_test_Heart_rate_reached_safety_level",
        "6032_raw-Maximum_workload_during_fitness_test",
        "6033_raw-Maximum_heart_rate_during_fitness_test",
        "6038_raw-Number_of_trend_entries",
        "6039-Duration_of_fitness_test",
        "ability to walk or cycle unaided for 10 minutes, self-reported",
        "achievement of target heart rate, self-reported",
    ]
] = "ECG"

trait_labels.loc[
    [
        "30000_raw-White_blood_cell_leukocyte_count",
        "30120_raw-Lymphocyte_count",
        "30130_raw-Monocyte_count",
        "30140_raw-Neutrophill_count",
        "30150-Eosinophill_count",
        "30180_raw-Lymphocyte_percentage",
        "30190_raw-Monocyte_percentage",
        "30200_raw-Neutrophill_percentage",
        "30210_raw-Eosinophill_percentage",
        "eosinophil count",
        "granulocyte count",
        "leukocyte count",
        "lymphocyte count",
        "monocyte count",
        "myeloid white cell count",
        "neutrophil count",
    ]
] = "White blood cells"


trait_labels.loc[
    [
        "30010_raw-Red_blood_cell_erythrocyte_count",
        "30020_raw-Haemoglobin_concentration",
        "30030_raw-Haematocrit_percentage",
        "30040_raw-Mean_corpuscular_volume",
        "30050_raw-Mean_corpuscular_haemoglobin",
        "30060_raw-Mean_corpuscular_haemoglobin_concentration",
        "30070_raw-Red_blood_cell_erythrocyte_distribution_width",
        "30240_raw-Reticulocyte_percentage",
        "30250_raw-Reticulocyte_count",
        "30260_raw-Mean_reticulocyte_volume",
        "30270_raw-Mean_sphered_cell_volume",
        "30280_raw-Immature_reticulocyte_fraction",
        "30290_raw-High_light_scatter_reticulocyte_percentage",
        "30300_raw-High_light_scatter_reticulocyte_count",
        "erythrocyte count",
        "reticulocyte count",
    ]
] = "Red blood cells"

trait_labels.loc[
    [
        "20015_raw-Sitting_height",
        "21001_raw-Body_mass_index_BMI",
        "21002_raw-Weight",
        "23098_raw-Weight",
        "23099_raw-Body_fat_percentage",
        "23100_raw-Whole_body_fat_mass",
        "23101_raw-Whole_body_fatfree_mass",
        "23102_raw-Whole_body_water_mass",
        "23104_raw-Body_mass_index_BMI",
        "23105_raw-Basal_metabolic_rate",
        "23111_raw-Leg_fat_percentage_right",
        "23112_raw-Leg_fat_mass_right",
        "23113_raw-Leg_fatfree_mass_right",
        "23114_raw-Leg_predicted_mass_right",
        "23115_raw-Leg_fat_percentage_left",
        "23116_raw-Leg_fat_mass_left",
        "23117_raw-Leg_fatfree_mass_left",
        "23118_raw-Leg_predicted_mass_left",
        "23119_raw-Arm_fat_percentage_right",
        "23120_raw-Arm_fat_mass_right",
        "23121_raw-Arm_fatfree_mass_right",
        "23122_raw-Arm_predicted_mass_right",
        "23123_raw-Arm_fat_percentage_left",
        "23124_raw-Arm_fat_mass_left",
        "23125_raw-Arm_fatfree_mass_left",
        "23126_raw-Arm_predicted_mass_left",
        "23127_raw-Trunk_fat_percentage",
        "23128_raw-Trunk_fat_mass",
        "23129_raw-Trunk_fatfree_mass",
        "23130_raw-Trunk_predicted_mass",
        "48_raw-Waist_circumference",
        "49_raw-Hip_circumference",
        "50_raw-Standing_height",
        "body height",
    ]
] = "Anthropometry"

trait_labels.loc[
    [
        "20003_1140861958-Treatmentmedication_code_simvastatin",
        "20003_1140868226-Treatmentmedication_code_aspirin",
        "20003_1140879802-Treatmentmedication_code_amlodipine",
        "20003_1141194794-Treatmentmedication_code_bendroflumethiazide",
        "4079_raw-Diastolic_blood_pressure_automated_reading",
        "4080_raw-Systolic_blood_pressure_automated_reading",
        "6150_1-Vascularheart_problems_diagnosed_by_doctor_Heart_attack",
        "6150_100-Vascularheart_problems_diagnosed_by_doctor_None_of_the_above",
        "6150_2-Vascularheart_problems_diagnosed_by_doctor_Angina",
        "6150_4-Vascularheart_problems_diagnosed_by_doctor_High_blood_pressure",
        "6153_1-Medication_for_cholesterol_blood_pressure_diabetes_or_take_exogenous_hormones_Cholesterol_lowering_medication",
        "6153_100-Medication_for_cholesterol_blood_pressure_diabetes_or_take_exogenous_hormones_None_of_the_above",
        "6153_2-Medication_for_cholesterol_blood_pressure_diabetes_or_take_exogenous_hormones_Blood_pressure_medication",
        "6154_1-Medication_for_pain_relief_constipation_heartburn_Aspirin",
        "6177_1-Medication_for_cholesterol_blood_pressure_or_diabetes_Cholesterol_lowering_medication",
        "6177_100-Medication_for_cholesterol_blood_pressure_or_diabetes_None_of_the_above",
        "6177_2-Medication_for_cholesterol_blood_pressure_or_diabetes_Blood_pressure_medication",
        "I9_CHD-Major_coronary_heart_disease_event",
        "I9_CHD_NOREV-Major_coronary_heart_disease_event_excluding_revascularizations",
        "I9_CORATHER-Coronary_atherosclerosis",
        "I9_IHD-Ischaemic_heart_disease_wide_definition",
        "I9_MI-Myocardial_infarction",
        "I9_MI_STRICT-Myocardial_infarction_strict",
        "I9_UAP-Unstable_angina_pectoris",
        "IX_CIRCULATORY-Diseases_of_the_circulatory_system",
        "acute myocardial infarction",
        "angina pectoris",
        "coronary artery disease",
        "hypercholesterolemia",
        "hypertension",
        "myocardial infarction",
    ]
] = "Cardiovascular"

trait_labels.loc[
    [
        "1717-Skin_colour",
        "1727-Ease_of_skin_tanning",
        "1737-Childhood_sunburn_occasions",
        "1747_1-Hair_colour_natural_before_greying_Blonde",
        "1747_2-Hair_colour_natural_before_greying_Red",
        "1747_3-Hair_colour_natural_before_greying_Light_brown",
        "1747_4-Hair_colour_natural_before_greying_Dark_brown",
        "1747_5-Hair_colour_natural_before_greying_Black",
        "2267-Use_of_sunuv_protection",
        "C3_SKIN-Malignant_neoplasm_of_skin",
        "C_MELANOMA_SKIN-Malignant_melanoma_of_skin",
        "C_OTHER_SKIN-Other_malignant_neoplasms_of_skin",
        "C_SKIN",
        "basal cell carcinoma",
        "skin neoplasm",
    ]
] = "Skin/hair"

# %% tags=[]
trait_labels

# %% tags=[]
clustrees_df = clustrees_df.assign(labels=trait_labels)

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(CONSENSUS_CLUSTERING_DIR, "clustering_tree_data.tsv").resolve()
display(output_file)

# %% tags=[]
clustrees_df.to_csv(output_file, sep="\t")

# %% tags=[]
