import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import copy


# Getting the current working directory
current_directory = os.getcwd()

# Constructing the relative path to the 'pilot trial data' folder
folder_path = os.path.join(current_directory, '2023-11-01_pilotTrial')

# Function to get the path to 'Outputs' folder
def find_folder(root_dir, name:str):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if name in dirnames:
            outputs_path = os.path.join(dirpath, name)
            return outputs_path
    return None

new_sampling_hz = 60
########################################## OpenCap ##########################################
opencap_hz = 60
# Get the path to the 'MarkerData' folder
opencap_markerdata_folder_path = find_folder(folder_path, 'MarkerData')

if opencap_markerdata_folder_path is not None:
    markerdata_header = pd.read_csv(os.path.join(opencap_markerdata_folder_path, 'treadmill_1.trc'), delimiter='\t',skiprows=3, nrows=2, header=None)
    # Getting header columns
    for i in range(1, len(markerdata_header.columns)):
        if pd.isna(markerdata_header.iloc[0,i]):
            markerdata_header.iloc[0,i] = markerdata_header.iloc[0,i-1]
    
    markerdata_header = markerdata_header.iloc[0] + '_' + markerdata_header.iloc[1] # Combine 2nd row with 1st row
    markerdata_header[0] = "Frame" # Rename first series
    markerdata_header[1] = "time" # Rename second series
    markerdata_header = markerdata_header[:-1] # Remove last row

    opencap_markerdata = pd.read_csv(os.path.join(opencap_markerdata_folder_path, 'treadmill_1.trc'), delimiter='\t', skiprows=4, on_bad_lines='skip').iloc[:,:-1]

    opencap_markerdata.columns = markerdata_header
else:
    print('MarkerData folder not found')

# # Calculate acceleration
# dt = opencap_markerdata['time'].diff()
# opencap_ankle_velocity = opencap_markerdata['r_ankle_study_X27'].diff() / dt
# opencap_ankle_acceleration = opencap_ankle_velocity.diff() / dt

# Upsample acceleration from 60Hz to 100Hz
# opencap_ankle_acceleration.fillna(0, inplace=True)

def calculate_new_sample(hz, df, new_sampling_hz):
    time = (1/hz) * len(df)
    number = int(new_sampling_hz * time)
    return number
# new_number_of_samples = calculate_new_sample(opencap_hz, opencap_ankle_acceleration)

# opencap_ankle_acceleration_resample = signal.resample(opencap_ankle_acceleration, new_number_of_samples)

# Get the path to the 'Outputs' folder
outputs_folder_path = find_folder(folder_path, 'Outputs')

if outputs_folder_path is not None:
    files_in_outputs = os.listdir(outputs_folder_path) # Get the list of files in the 'Outputs' folder
else:
    print('Outputs folder not found')

# Loop through the files in the 'Outputs' folder to search for the angle data
for file_name in files_in_outputs:
    if 'values' in file_name.lower() and file_name.endswith('.csv'):
        file_path = os.path.join(outputs_folder_path, file_name)
        opencapData_angle = pd.read_csv(file_path)
        opencapData_angle = opencapData_angle.drop(opencapData_angle.columns[0], axis=1)

# # Resampling OpenCap
# # Apply the resampling to each column in the DataFrame and store the results in a list
# resampled_columns = []
# for column_name in opencapData_angle.columns:
#     original = opencapData_angle[column_name]
#     new_number_of_samples = calculate_new_sample(opencap_hz, original, new_sampling_hz)
#     resampled = pd.Series(signal.resample(original, new_number_of_samples), name=column_name)
#     resampled_columns.append(resampled)

# # Concatenate all resampled columns into a new DataFrame
# opencapData_angle_resampled = pd.concat(resampled_columns, axis=1)

opencapData_angle_resampled = opencapData_angle

# Index(['time', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx',
#        'pelvis_ty', 'pelvis_tz', 'hip_flexion_r', 'hip_adduction_r',
#        'hip_rotation_r', 'knee_angle_r', 'knee_angle_r_beta', 'ankle_angle_r',
#        'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l',
#        'hip_rotation_l', 'knee_angle_l', 'knee_angle_l_beta', 'ankle_angle_l',
#        'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension', 'lumbar_bending',
#        'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
#        'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
#        'elbow_flex_l', 'pro_sup_l', 'dtime'],
#       dtype='object')

########################################## 3D Mocap ##########################################
mocap_hz = 200

mocap_folder_path = find_folder(folder_path, 'mocap')

mocap_markerdata = pd.read_csv(os.path.join(mocap_folder_path, 'Run 1 - Copy.csv'), skiprows=49602)
mocap_markerdata.drop(mocap_markerdata.columns[0], axis=1, inplace=True)
# Todo: Drop column 0,1,15:62 from the original dataframe

namelist = []
for i in range(1, len(mocap_markerdata.columns)):
    name = mocap_markerdata.iloc[0,i] + "_" + mocap_markerdata.iloc[1,i]
    namelist.append(name)

mocap_markerdata = mocap_markerdata.iloc[3:9803,1:]
mocap_markerdata.columns = namelist
mocap_markerdata = mocap_markerdata.astype(float)

# Add time to mocap_markerdata
mocap_time = np.arange(0, len(mocap_markerdata) / mocap_hz, 1 / mocap_hz)
mocap_markerdata['time'] = mocap_time

# Downsample markerdata from 200Hz to new sampling rate
resampled_columns = []
for column_name in mocap_markerdata.columns:
    original = mocap_markerdata[column_name]
    new_number_of_samples = calculate_new_sample(mocap_hz, original, new_sampling_hz)  
    resampled = pd.Series(signal.resample(original, new_number_of_samples), name=column_name)
    resampled_columns.append(resampled)

# Concatenate all resampled columns into a new DataFrame
mocap_markerdata_resampled = pd.concat(resampled_columns, axis=1)

# Calculate acceleration
dt = mocap_markerdata_resampled['time'].diff()
mocap_toe_velocity = mocap_markerdata_resampled['R Toe_X'].diff() / dt
mocap_toe_acceleration = mocap_toe_velocity.diff() / dt

# Read mocap Kinematics
mocapKinematics = pd.read_csv(os.path.join(mocap_folder_path, 'Kinematics_12Hz.csv'), sep="\t")

namelist = []
for i in range(2, len(mocapKinematics.columns)):
    name = mocapKinematics.iloc[0,i] + "_" + mocapKinematics.iloc[3,i]
    namelist.append(name)

mocapKinematics = mocapKinematics.iloc[4:9804,2:]
mocapKinematics.columns = namelist
mocapKinematics = mocapKinematics.astype(float)

# Downsampling mocapKinematics
# Apply the resampling to each column in the DataFrame and store the results in a list
resampled_columns = []
for column_name in mocapKinematics.columns:
    original = mocapKinematics[column_name]
    new_number_of_samples = calculate_new_sample(mocap_hz, original, new_sampling_hz)  
    resampled = pd.Series(signal.resample(original, new_number_of_samples), name=column_name)
    resampled_columns.append(resampled)

# Concatenate all resampled columns into a new DataFrame
mocapKinematics_resampled = pd.concat(resampled_columns, axis=1)

# Updated using IK from OpenSim
updated_folder = find_folder(folder_path, 'updated names')
mocapKinematics_updated = pd.read_csv(os.path.join(updated_folder, 'Run 1_output_2_IK.mot'), delimiter='\t', skiprows=10)

resampled_columns = []
for column_name in mocapKinematics_updated.columns:
    original = mocapKinematics_updated[column_name]
    new_number_of_samples = calculate_new_sample(mocap_hz, original, new_sampling_hz)  
    resampled = pd.Series(signal.resample(original, new_number_of_samples), name=column_name)
    resampled_columns.append(resampled)
mocapKinematics_updated_resampled = pd.concat(resampled_columns, axis=1)

# print(f'Mocap markerdata len: {len(mocap_markerdata_resampled)}')
# print(f'Mocap kinematic len: {len(mocapKinematics_resampled)}')

# Index(['R_Ankle Virtual Foot_JOINT_ANGLE_X',
#        'R_Ankle Virtual Foot_JOINT_ANGLE_Y',
#        'R_Ankle Virtual Foot_JOINT_ANGLE_Z',
#        'L_Ankle Virtual Foot_JOINT_ANGLE_X',
#        'L_Ankle Virtual Foot_JOINT_ANGLE_Y',
#        'L_Ankle Virtual Foot_JOINT_ANGLE_Z', 'R_Hip_JOINT_ANGLE_X',
#        'R_Hip_JOINT_ANGLE_Y', 'R_Hip_JOINT_ANGLE_Z', 'L_Hip_JOINT_ANGLE_X',
#        'L_Hip_JOINT_ANGLE_Y', 'L_Hip_JOINT_ANGLE_Z', 'R_Knee_JOINT_ANGLE_X',
#        'R_Knee_JOINT_ANGLE_Y', 'R_Knee_JOINT_ANGLE_Z', 'L_Knee_JOINT_ANGLE_X',
#        'L_Knee_JOINT_ANGLE_Y', 'L_Knee_JOINT_ANGLE_Z'],
#       dtype='object')

########################################## IMU ##########################################
# IMU
imu_hz = 100

imuFile = 'imu_1_run_1.csv'
imuData = pd.read_csv(os.path.join(folder_path, imuFile), skiprows=3)
imuData = imuData.drop(['time','Activity','Marker'],axis=1)
# Index(['RT Foot course (deg)', 'RT Foot pitch (deg)', 'RT Foot roll (deg)',
    #    'RT Foot Tilt Fwd (deg)', 'RT Foot Tilt Med (deg)',
    #    'RT Foot Rotation Ext (deg)', 'LT Hip Flexion (deg)',
    #    'LT Hip Abduction (deg)', 'LT Hip Rotation Ext (deg)',
    #    'LT Knee Abduction (deg)', 'LT Knee Rotation Ext (deg)',
    #    'LT Knee Flexion (deg)',
    #    'Noraxon MyoMotion-Joints-Knee LT-Extension (deg)',
    #    'LT Ankle Dorsiflexion (deg)', 'LT Ankle Abduction (deg)',
    #    'LT Ankle Inversion (deg)', 'RT Hip Flexion (deg)',
    #    'RT Hip Abduction (deg)', 'RT Hip Rotation Ext (deg)',
    #    'RT Knee Abduction (deg)', 'RT Knee Rotation Ext (deg)',
    #    'RT Knee Flexion (deg)',
    #    'Noraxon MyoMotion-Joints-Knee RT-Extension (deg)',
    #    'RT Ankle Dorsiflexion (deg)', 'RT Ankle Abduction (deg)',
    #    'RT Ankle Inversion (deg)'],
    #   dtype='object')

# Filter data
cutoff_frequency = 12  # Hz
order = 4  # Filter order
b, a = signal.butter(order, cutoff_frequency/(imu_hz/2), btype='lowpass', output='ba')

# Apply the filter to each column in the DataFrame and store the results in a list
filtered_columns = []
for column_name in imuData.columns:
    original = imuData[column_name]
    filtered = pd.Series(signal.filtfilt(b, a, original), name=column_name)
    filtered_columns.append(filtered)

# Concatenate all filtered columns into a new DataFrame
imu_filtered = pd.concat(filtered_columns, axis=1)

# Downsampling IMU
# Apply the resampling to each column in the DataFrame and store the results in a list
resampled_columns = []
for column_name in imu_filtered.columns:
    original = imu_filtered[column_name]
    new_number_of_samples = calculate_new_sample(imu_hz, original, new_sampling_hz)  
    resampled = pd.Series(signal.resample(original, new_number_of_samples), name=column_name)
    resampled_columns.append(resampled)
# Concatenate all resampled columns into a new DataFrame
imu_filtered_resampled = pd.concat(resampled_columns, axis=1)

# Plot the original and filtered signals for 'RT Knee Flexion (deg)'
# plt.plot(imuData['RT Knee Flexion (deg)'])
# plt.plot(imu_filtered['RT Knee Flexion (deg)'])
# plt.legend(['Original', 'Filtered'])
# plt.show()

########################################## Time sync ONE ##########################################

# Using knee kinematics to find the max value then match
sig1 = mocapKinematics_resampled['R_Knee_JOINT_ANGLE_X'][:2000]
sig2 = imu_filtered_resampled['RT Knee Flexion (deg)'][:1000]
sig3 = opencapData_angle_resampled['knee_angle_r'][:1000]
sig4 = mocapKinematics_updated_resampled['knee_angle_r'][:1000]

sig1_max = np.argmax(sig1)
sig2_max = np.argmax(sig2)
sig3_max = np.argmax(sig3)
sig4_max = np.argmax(sig4)

# Go to kinematics data and cutoff the data before the max value
mocap_markerdata_resampled = mocap_markerdata_resampled.iloc[sig1_max:]
mocapKinematics_resampled = mocapKinematics_resampled.iloc[sig1_max:]
imu_filtered_resampled = imu_filtered_resampled.iloc[sig2_max:]
opencapData_angle_resampled = opencapData_angle_resampled.iloc[sig3_max:]
mocapKinematics_updated_resampled = mocapKinematics_updated_resampled.iloc[sig4_max:]

# Reset index for all
mocap_markerdata_resampled.reset_index(inplace=True)
mocapKinematics_resampled.reset_index(inplace=True)
imu_filtered_resampled.reset_index(inplace=True)
opencapData_angle_resampled.reset_index(inplace=True)
mocapKinematics_updated_resampled.reset_index(inplace=True)

# get the smallest length of the 3 data, so that all is equal
max_time_cutoff = min(len(mocapKinematics_resampled),len(imu_filtered_resampled),len(opencapData_angle_resampled))
# max_time_cutoff_2 = min(len(mocapKinematics_updated_resampled),len(imu_filtered_resampled),len(opencapData_angle_resampled)) # Same as above 2126

# use the max_time_cutoff to trim the data
mocap_markerdata_resampled = mocap_markerdata_resampled.iloc[:max_time_cutoff]
mocapKinematics_resampled = mocapKinematics_resampled.iloc[:max_time_cutoff]
imu_filtered_resampled = imu_filtered_resampled.iloc[:max_time_cutoff]
opencapData_angle_resampled = opencapData_angle_resampled.iloc[:max_time_cutoff]
mocapKinematics_updated_resampled = mocapKinematics_updated_resampled.iloc[:max_time_cutoff]

# Rename columns
# Define the mapping of old column names to new names
column_mapping = {
    'mocap': {
        'R_Hip_JOINT_ANGLE_X': 'R_Hip_Flexion',
        'R_Knee_JOINT_ANGLE_X': 'R_Knee_Flexion',
        'R_Ankle Virtual Foot_JOINT_ANGLE_X': 'R_Ankle_Flexion',
        'L_Hip_JOINT_ANGLE_X': 'L_Hip_Flexion',
        'L_Knee_JOINT_ANGLE_X': 'L_Knee_Flexion',
        'L_Ankle Virtual Foot_JOINT_ANGLE_X': 'L_Ankle_Flexion',
    },
    'imu': {
        'RT Hip Flexion (deg)': 'R_Hip_Flexion',
        'RT Knee Flexion (deg)': 'R_Knee_Flexion',
        'RT Ankle Dorsiflexion (deg)': 'R_Ankle_Flexion',
        'LT Hip Flexion (deg)': 'L_Hip_Flexion',
        'LT Knee Flexion (deg)': 'L_Knee_Flexion',
        'LT Ankle Dorsiflexion (deg)': 'L_Ankle_Flexion',
    },
    'opencap': {
        'hip_flexion_r': 'R_Hip_Flexion',
        'knee_angle_r': 'R_Knee_Flexion',
        'ankle_angle_r': 'R_Ankle_Flexion',
        'hip_flexion_l': 'L_Hip_Flexion',
        'knee_angle_l': 'L_Knee_Flexion',
        'ankle_angle_l': 'L_Ankle_Flexion',
    },
    'mocap_updated': {
        'hip_flexion_r': 'R_Hip_Flexion',
        'knee_angle_r': 'R_Knee_Flexion',
        'ankle_angle_r': 'R_Ankle_Flexion',
        'hip_flexion_l': 'L_Hip_Flexion',
        'knee_angle_l': 'L_Knee_Flexion',
        'ankle_angle_l': 'L_Ankle_Flexion',
    }
}

# Rename the columns in each DataFrame
mocapKinematics_resampled.rename(columns=column_mapping['mocap'], inplace=True)
imu_filtered_resampled.rename(columns=column_mapping['imu'], inplace=True)
opencapData_angle_resampled.rename(columns=column_mapping['opencap'], inplace=True)
mocapKinematics_updated_resampled.rename(columns=column_mapping['mocap_updated'], inplace=True)

########################################## Time sync TWO ##########################################

# Using mocap right knee, identified where gait cycle first occured using peak.

# Find the peaks in the data
# data = mocapKinematics_resampled['R_Knee_Flexion']
# peak_indices, _ = find_peaks(data, height=60, distance=50)

data = mocapKinematics_updated_resampled['R_Knee_Flexion']
peak_indices, _ = find_peaks(data, height=60, distance=50)

# start_index = peak_indices[0] # using mocapKinematics_resampled
start_index = peak_indices[2] # using mocapKinematics_updated_resampled

mocap_markerdata_resampled = mocap_markerdata_resampled[start_index:].reset_index(drop=True)
mocapKinematics_resampled = mocapKinematics_resampled[start_index:].reset_index(drop=True)
imu_filtered_resampled = imu_filtered_resampled[start_index:].reset_index(drop=True)
opencapData_angle_resampled = opencapData_angle_resampled[start_index:].reset_index(drop=True)
mocapKinematics_updated_resampled = mocapKinematics_updated_resampled[start_index:].reset_index(drop=True)

########################################## Mean absolute error and mean absolute percentage error ##########################################
from sklearn.metrics import mean_absolute_error, mean_squared_error
joint_angles = ['Hip_Flexion', 'Knee_Flexion', 'Ankle_Flexion']
sides = ['L', 'R']
datasets = ['mocap', 'imu', 'opencap', 'mocap_updated']
dataframes_dict = {'mocap': mocapKinematics_resampled, 'imu': imu_filtered_resampled, 'opencap': opencapData_angle_resampled, 'mocap_updated': mocapKinematics_updated_resampled}

# Initialize dictionaries to store the MAE and MAPE values
mae_values = {}
rmse_values = {}

# Iterate over the datasets
for i, dataset1 in enumerate(datasets):
    for dataset2 in datasets[i+1:]:
        # Iterate over the joint angles and sides
        for joint_angle in joint_angles:
            for side in sides:
                # print(f'dataset1: {dataset1}')
                # print(f'dataset2: {dataset2}')
                # Calculate the MAE and MAPE between the two datasets for this joint angle and side
                key = f'{dataset1}_vs_{dataset2}_{side}_{joint_angle}'
                y_true = dataframes_dict[dataset1][f'{side}_{joint_angle}']
                y_pred = dataframes_dict[dataset2][f'{side}_{joint_angle}']
                mae_values[key] = mean_absolute_error(y_true, y_pred)
                # mape_values[key] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                rmse_values[key] = np.sqrt(mean_squared_error(y_true, y_pred))

# Create a DataFrame from the MAE, MAPE, and RMSE dictionaries
error_df = pd.DataFrame({
    'MAE': mae_values,
    'RMSE': rmse_values
})

print('before correction')
print(error_df)

# Save the DataFrame to an Excel file with a specific sheet name
# with pd.ExcelWriter(f'dataframes_resample_{new_sampling_hz}hz.xlsx') as writer:
#     mocapKinematics_resampled.to_excel(writer, sheet_name='mocap')
#     imu_filtered.to_excel(writer, sheet_name='imu')
#     opencapData_angle_resampled.to_excel(writer, sheet_name='opencap')

# ########################################## Plotting before correction ##########################################
# Define the angles and sides
angles = ['Hip_Flexion', 'Knee_Flexion', 'Ankle_Flexion']
sides = ['L', 'R']

# # Create a 3x2 grid of subplots
# fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# # Set a title for the entire figure
# fig.suptitle(f'Kinematic Angles Comparison ({new_sampling_hz}hz)', fontsize=16)

# # Iterate over the sides and angles
# for i, angle in enumerate(angles):
#     for j, side in enumerate(sides):
#         # Create the kinematic angle string
#         kinematic_angle = f'{side}_{angle}'

#         # Plot the data on the appropriate subplot
#         axs[i, j].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000])
#         axs[i, j].plot(imu_filtered_resampled[kinematic_angle][:1000])
#         axs[i, j].plot(opencapData_angle_resampled[kinematic_angle][:1000])
#         axs[i, j].legend(['Marker-based', 'IMU', 'OpenCap'])
#         axs[i, j].set_title(kinematic_angle)

# # Display the plots
# plt.tight_layout()
# # Save the figure to a file
# # plt.savefig(f'Kinematic_angles_comparison_{new_sampling_hz}hz_before correction.png')
# plt.show()

# ########################################## Do some correction ##########################################
# Psuedocode:
# Based on the first index, check whats the difference between IMU/OpenCap with VICON mocap
# If it's under VICON mocap, then add the difference to the IMU/OpenCap data for all
# If it's above VICON mocap, then subtract the difference to the IMU/OpenCap data for all

# Define the datasets
datasets = ['imu', 'opencap']
mocap_dataset = 'mocap_updated'

mocapKinematics_resampled_correction = copy.deepcopy(mocapKinematics_resampled)
imu_filtered_resampled_correction = copy.deepcopy(imu_filtered_resampled)
opencapData_angle_resampled_correction = copy.deepcopy(opencapData_angle_resampled)
mocapKinematics_updated_resampled_correction = copy.deepcopy(mocapKinematics_updated_resampled)

dataframes_dict_after = {'mocap': mocapKinematics_resampled_correction, 'imu': imu_filtered_resampled_correction, 'opencap': opencapData_angle_resampled_correction, 'mocap_updated': mocapKinematics_updated_resampled_correction}

# Iterate over the datasets
for dataset in datasets:
    # Iterate over the joint angles and sides
    for joint_angle in joint_angles:
        for side in sides:
            # Calculate the difference at the first index
            difference = dataframes_dict_after[dataset][f'{side}_{joint_angle}'].iloc[0] - dataframes_dict_after[mocap_dataset][f'{side}_{joint_angle}'].iloc[0]

            # If the difference is positive, subtract it from the entire dataset
            # If the difference is negative, add its absolute value to the entire dataset
            if difference > 0:
                dataframes_dict_after[f'{dataset}'][f'{side}_{joint_angle}'] = dataframes_dict_after[dataset][f'{side}_{joint_angle}'] - difference
            else:
                dataframes_dict_after[f'{dataset}'][f'{side}_{joint_angle}'] = dataframes_dict_after[dataset][f'{side}_{joint_angle}'] + abs(difference)

########################################## Calculate error after correction ##########################################
datasets = ['mocap', 'imu', 'opencap', 'mocap_updated']
# dataframes_dict = {'mocap': mocapKinematics_resampled, 'imu': imu_filtered_resampled, 'opencap': opencapData_angle_resampled}

# Initialize dictionaries to store the MAE and MAPE values
mae_values = {}
rmse_values = {}

# Iterate over the datasets
for i, dataset1 in enumerate(datasets):
    for dataset2 in datasets[i+1:]:
        # Iterate over the joint angles and sides
        for joint_angle in joint_angles:
            for side in sides:
                # print(f'dataset1: {dataset1}')
                # print(f'dataset2: {dataset2}')
                # Calculate the MAE and MAPE between the two datasets for this joint angle and side
                key = f'{dataset1}_vs_{dataset2}_{side}_{joint_angle}'
                y_true = dataframes_dict_after[dataset1][f'{side}_{joint_angle}']
                y_pred = dataframes_dict_after[dataset2][f'{side}_{joint_angle}']
                mae_values[key] = mean_absolute_error(y_true, y_pred)
                # mape_values[key] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                rmse_values[key] = np.sqrt(mean_squared_error(y_true, y_pred))

# Create a DataFrame from the MAE, MAPE, and RMSE dictionaries
correction_error_df = pd.DataFrame({
    'MAE': mae_values,
    'RMSE': rmse_values
})

print('after correction')
print(correction_error_df)

# Save the DataFrame to an Excel file with a specific sheet name
with pd.ExcelWriter(f'error_metrics_{new_sampling_hz}hz.xlsx') as writer:
    error_df.to_excel(writer, sheet_name=f'resample_{new_sampling_hz}hz_before')
    correction_error_df.to_excel(writer, sheet_name=f'resample_{new_sampling_hz}hz_after')

# ########################################## Plotting after correction ##########################################
# Create a 3x2 grid of subplots
# fig, axs = plt.subplots(3, 2, figsize=(10, 15))

# # Set a title for the entire figure
# fig.suptitle(f'Kinematic Angles Comparison ({new_sampling_hz}hz after correction)', fontsize=16)

# # Iterate over the sides and angles
# for i, angle in enumerate(angles):
#     for j, side in enumerate(sides):
#         # Create the kinematic angle string
#         kinematic_angle = f'{side}_{angle}'

#         # Plot the data on the appropriate subplot
#         axs[i, j].plot(mocapKinematics_resampled_correction[kinematic_angle][:1000])
#         axs[i, j].plot(imu_filtered_resampled_correction[kinematic_angle][:1000])
#         axs[i, j].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000])
#         axs[i, j].legend(['VICON', 'IMU', 'OpenCap'])
#         axs[i, j].set_title(kinematic_angle)

# Display the plots
# plt.tight_layout()
# Save the figure to a file
# plt.savefig(f'Kinematic_angles_comparison_{new_sampling_hz}hz after correction.png')
# plt.show()

# with pd.ExcelWriter(f'dataframes_resample_{new_sampling_hz}hz_after correction.xlsx') as writer:
#     mocapKinematics_resampled.to_excel(writer, sheet_name='mocap')
#     imu_filtered.to_excel(writer, sheet_name='imu')
#     opencapData_angle_resampled.to_excel(writer, sheet_name='opencap')

# ########################################## Plotting before and after correction ##########################################
angles = ['Hip_Flexion', 'Knee_Flexion', 'Ankle_Flexion']
sides = ['L', 'R']

fig, axs = plt.subplots(6, 2, figsize=(10, 15))

# Set a title for the entire figure
# fig.suptitle(f'Kinematic Angles Comparison', fontsize=16)

# Plot the data on the appropriate subplot
kinematic_angle = "L_Hip_Flexion"
axs[0, 0].text(-100, 70, "A", size=18, weight='bold')
l2, = axs[0, 0].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000], color='green')
l3, = axs[0, 0].plot(imu_filtered_resampled[kinematic_angle][:1000], alpha=0.7, color='blue')
l4, = axs[0, 0].plot(opencapData_angle_resampled[kinematic_angle][:1000], alpha=0.5, color='red')
# axs[i, j].legend(['VICON', 'IMU', 'OpenCap'])
axs[0, 0].set_title("L Hip Flexion")
axs[0, 0].set_xlabel('Frame (n)', fontsize=10)
axs[0, 0].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "R_Hip_Flexion"
axs[1, 0].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000], color='green')
axs[1, 0].plot(imu_filtered_resampled[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[1, 0].plot(opencapData_angle_resampled[kinematic_angle][:1000], alpha=0.5, color='red')
# axs[i, j].legend(['VICON', 'IMU', 'OpenCap'])
axs[1, 0].set_title("R Hip Flexion")
axs[1, 0].set_xlabel('Frame (n)', fontsize=10)
axs[1, 0].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "L_Knee_Flexion"
axs[2, 0].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000], color='green')
axs[2, 0].plot(imu_filtered_resampled[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[2, 0].plot(opencapData_angle_resampled[kinematic_angle][:1000], alpha=0.5, color='red')
axs[2, 0].set_title("L Knee Flexion")
axs[2, 0].set_xlabel('Frame (n)', fontsize=10)
axs[2, 0].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "R_Knee_Flexion"
axs[3, 0].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000], color='green')
axs[3, 0].plot(imu_filtered_resampled[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[3, 0].plot(opencapData_angle_resampled[kinematic_angle][:1000], alpha=0.5, color='red')
axs[3, 0].set_title("R Knee Flexion")
axs[3, 0].set_xlabel('Frame (n)', fontsize=10)
axs[3, 0].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "L_Ankle_Flexion"
axs[4, 0].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000], color='green')
axs[4, 0].plot(imu_filtered_resampled[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[4, 0].plot(opencapData_angle_resampled[kinematic_angle][:1000], alpha=0.5, color='red')
axs[4, 0].set_title("L Ankle Flexion")
axs[4, 0].set_xlabel('Frame (n)', fontsize=10)
axs[4, 0].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "R_Ankle_Flexion"
axs[5, 0].plot(mocapKinematics_updated_resampled[kinematic_angle][:1000], color='green')
axs[5, 0].plot(imu_filtered_resampled[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[5, 0].plot(opencapData_angle_resampled[kinematic_angle][:1000], alpha=0.5, color='red')
axs[5, 0].set_title("R Ankle Flexion")
axs[5, 0].set_xlabel('Frame (n)', fontsize=10)
axs[5, 0].set_ylabel('Degree (°)', fontsize=10)

# Correction
kinematic_angle = "L_Hip_Flexion"
axs[0, 1].text(-100, 70, "B", size=18, weight='bold')
axs[0, 1].plot(mocapKinematics_updated_resampled_correction[kinematic_angle][:1000], color='green')
axs[0, 1].plot(imu_filtered_resampled_correction[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[0, 1].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000], alpha=0.5, color='red')
# axs[i, j].legend(['VICON', 'IMU', 'OpenCap'])
axs[0, 1].set_title("L Hip Flexion")
axs[0, 1].set_xlabel('Frame (n)', fontsize=10)
axs[0, 1].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "R_Hip_Flexion"
axs[1, 1].plot(mocapKinematics_updated_resampled_correction[kinematic_angle][:1000], color='green')
axs[1, 1].plot(imu_filtered_resampled_correction[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[1, 1].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000], alpha=0.5, color='red')
# axs[i, j].legend(['VICON', 'IMU', 'OpenCap'])
axs[1, 1].set_title("R Hip Flexion")
axs[1, 1].set_xlabel('Frame (n)', fontsize=10)
axs[1, 1].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "L_Knee_Flexion"
axs[2, 1].plot(mocapKinematics_updated_resampled_correction[kinematic_angle][:1000], color='green')
axs[2, 1].plot(imu_filtered_resampled_correction[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[2, 1].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000], alpha=0.5, color='red')
axs[2, 1].set_title("L Knee Flexion")
axs[2, 1].set_xlabel('Frame (n)', fontsize=10)
axs[2, 1].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "R_Knee_Flexion"
axs[3, 1].plot(mocapKinematics_updated_resampled_correction[kinematic_angle][:1000], color='green')
axs[3, 1].plot(imu_filtered_resampled_correction[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[3, 1].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000], alpha=0.5, color='red')
axs[3, 1].set_title("R Knee Flexion")
axs[3, 1].set_xlabel('Frame (n)', fontsize=10)
axs[3, 1].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "L_Ankle_Flexion"
axs[4, 1].plot(mocapKinematics_updated_resampled_correction[kinematic_angle][:1000], color='green')
axs[4, 1].plot(imu_filtered_resampled_correction[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[4, 1].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000], alpha=0.5, color='red')
axs[4, 1].set_title("L Ankle Flexion")
axs[4, 1].set_xlabel('Frame (n)', fontsize=10)
axs[4, 1].set_ylabel('Degree (°)', fontsize=10)

kinematic_angle = "R_Ankle_Flexion"
axs[5, 1].plot(mocapKinematics_updated_resampled_correction[kinematic_angle][:1000], color='green')
axs[5, 1].plot(imu_filtered_resampled_correction[kinematic_angle][:1000], alpha=0.7, color='blue')
axs[5, 1].plot(opencapData_angle_resampled_correction[kinematic_angle][:1000], alpha=0.5, color='red')
axs[5, 1].set_title("R Ankle Flexion")
axs[5, 1].set_xlabel('Frame (n)', fontsize=10)
axs[5, 1].set_ylabel('Degree (°)', fontsize=10)
 
fig.legend((l2, l3, l4), ('Marker-based','IMU',"OpenCap"), loc = 'lower center', bbox_to_anchor=(0.5, -0.02), ncol=3)

# Display the plots
plt.tight_layout()
# Save the figure to a file
# plt.savefig(f'Kinematic_angles_comparison_both.png')
plt.show()

################################################## Function to plot ##################################################
# import matplotlib.pyplot as plt

# def plot_kinematic_angle(ax, kinematic_angle, mocap_data, imu_data, opencap_data, label):
#     ax.plot(mocap_data[kinematic_angle][:1000], color='green')
#     ax.plot(imu_data[kinematic_angle][:1000], alpha=0.7, color='blue')
#     ax.plot(opencap_data[kinematic_angle][:1000], alpha=0.5, color='red')
#     ax.set_title(label)
#     ax.set_xlabel('Frame (n)', fontsize=10)
#     ax.set_ylabel('Degree (°)', fontsize=10)

# angles = ['Hip_Flexion', 'Knee_Flexion', 'Ankle_Flexion']
# sides = ['L', 'R']

# fig, axs = plt.subplots(6, 2, figsize=(10, 15))

# # Plot original data
# for i, side in enumerate(sides):
#     for j, angle in enumerate(angles):
#         kinematic_angle = f"{side}_{angle}"
#         label = f"{side} {angle.replace('_', ' ')}"
#         plot_kinematic_angle(axs[j + i * len(angles), 0], kinematic_angle,
#                               mocapKinematics_updated_resampled, imu_filtered_resampled, opencapData_angle_resampled, label)

# # Plot corrected data
# for i, side in enumerate(sides):
#     for j, angle in enumerate(angles):
#         kinematic_angle = f"{side}_{angle}"
#         label = f"{side} {angle.replace('_', ' ')}"
#         plot_kinematic_angle(axs[j + i * len(angles), 1], kinematic_angle,
#                               mocapKinematics_updated_resampled_correction, imu_filtered_resampled_correction,
#                               opencapData_angle_resampled_correction, label)

# fig.legend(('Marker-based', 'IMU', 'OpenCap'), loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3)

# # Display the plots
# plt.tight_layout()
# # Save the figure to a file
# # plt.savefig('Kinematic_angles_comparison_both.png')
# plt.show()