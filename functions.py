import os
import matplotlib.pyplot as plt

# Function to get the path to 'Outputs' folder
def find_folder(root_dir, name:str):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if name in dirnames:
            outputs_path = os.path.join(dirpath, name)
            return outputs_path
    return None

def calculate_new_sample(hz, df, new_sampling_hz):
    time = (1/hz) * len(df)
    number = int(new_sampling_hz * time)
    return number

def plot_all_kinematics(plotname:str, dataset_mocap, imu_dataset, opencap_dataset):
    angles = ['Hip_Flexion', 'Knee_Flexion', 'Ankle_Flexion']
    sides = ['L', 'R']

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle(f'{plotname}', fontsize=16)
    for i, angle in enumerate(angles):
        for j, side in enumerate(sides):
            # Create the kinematic angle string
            kinematic_angle = f'{side}_{angle}'

            # Plot the data on the appropriate subplot
            axs[i, j].plot(dataset_mocap[kinematic_angle][:1000])
            axs[i, j].plot(imu_dataset[kinematic_angle][:1000])
            axs[i, j].plot(opencap_dataset[kinematic_angle][:1000])
            axs[i, j].legend(['VICON', 'IMU', 'OpenCap'])
            axs[i, j].set_title(kinematic_angle)

    plt.tight_layout()

    # Save the figure to a file
    # plt.savefig(f'{plotname}.png')
    plt.show()
