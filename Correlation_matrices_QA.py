import os
import numpy as np
import pandas as pd
from nilearn import plotting
import scipy
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt


# fetch atlas
atlas=datasets.fetch_atlas_difumo(dimension=256, resolution_mm=2, data_dir=None, resume=True, verbose=1, legacy_format=False)
#atlas = datasets.fetch_atlas_msdl()  #previous atlas
# Loading atlas image stored in 'maps'
atlas_filename = atlas["maps"]
# Loading atlas data stored in 'labels'
labels = atlas["labels"]
print("Atlas file path:", atlas_filename)

# Initialize the masker
masker = NiftiMapsMasker(
    maps_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=0, #1 if you want basic progress information
    tr=2.5
)

# Prepare an empty array to store the correlation matrices (shape: (76 subjects, 256 regions, 256 regions))
cor_mat_array_76 = np.full((77, 256, 256), np.nan)

#subject_idx = 0  # Index for storing results in cor_mat_array_76
# List to store the IDs of excluded subjects
excluded_subjects = [] 
#creating an array for time series
shape = (77, 295, 256)
time_series_arr = np.full(shape, np.nan)

for subject_id in range(1, 77):
    # Define the paths to the functional and confound files
    func_file = f'/home/ptslab/Pain_Data/preproc/sub-{subject_id:02d}/func/sub-{subject_id:02d}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    confound_file = f'/home/ptslab/Pain_Data/preproc/sub-{subject_id:02d}/func/sub-{subject_id:02d}_task-rest_desc-confounds_timeseries.tsv'
    
    # Load the confound file
    confound = pd.read_csv(confound_file, sep="\t", na_values="n/a")
     # Check if framewise displacement (FD) > 0.6 for any frame; if yes, skip the subject
    fd = confound['framewise_displacement']
    if fd.mean() > 0.6:  # If avg FD exceeds 0.6, skip this subject
        cor_mat_array_76[subject_id, :, :] = np.nan #put nan in all values of subjects with FD>0.6
        excluded_subjects.append(subject_id) 
        print(f"Skipping subject {subject_id} due to FD > 0.6")
        
        continue

      # Select relevant columns and fill missing values with 0
    finalConf = confound[['csf', 'white_matter', 'framewise_displacement',
                          'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04',
                          'a_comp_cor_05', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]
    finalConf = finalConf.fillna(0)
    
    # Preprocess the functional file and extract the time series
    time_series = masker.fit_transform(func_file, confounds=finalConf)
    
    # Remove the first 5 frames from the time series
    time_series = time_series[5:]
    time_series_arr[subject_id, :time_series.shape[0], :time_series.shape[1]] = time_series
    # Create a ConnectivityMeasure object to compute correlation
    correlation_measure = ConnectivityMeasure(kind='correlation')
    
    # Compute the correlation matrix
    cor_mat = correlation_measure.fit_transform([time_series])[0]
    
    # Store the correlation matrix in the array
    cor_mat_array_76[subject_id, :, :] = cor_mat
    #subject_idx += 1
# Save files
np.save(r'/home/ptslab/Pain_Data/Pain_OpenNeuro/Arrays/time_series_data.npy', time_series_arr)
np.save(r'/home/ptslab/Pain_Data/Pain_OpenNeuro/Arrays/corr_mat_76.npy',cor_mat_array_76)
np.save(r'/home/ptslab/Pain_Data/OpenNeuro_pain_Data/excluded_subjects/Excluded subjects.npy', excluded_subjects)
# Save list of excluded to a txt file:
output_txt_path= r'/home/ptslab/Pain_Data/OpenNeuro_pain_Data/excluded_subjects/Excluded_subjects.csv'
excluded_df = pd.DataFrame({'Excluded_Subjects': excluded_subjects})
excluded_df.to_csv(output_txt_path, index=False, sep="\t")  # Use tab-separated values for clarity
    
