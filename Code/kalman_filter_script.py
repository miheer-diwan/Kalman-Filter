import numpy as np
import matplotlib.pyplot as plt
from kalman_filter_functions import load_data, get_del_t, get_parameters, kalman_filter

# Plot titles corresponding to each filename
plot_titles = [
    "Motion Capture",
    "Low Noise",
    "High Noise",
    "Velocity"
]

filenames = [
    "project_2_kalman_filter\Data\kalman_filter_data_mocap.txt",
    "project_2_kalman_filter\Data\kalman_filter_data_low_noise.txt",
    "project_2_kalman_filter\Data\kalman_filter_data_high_noise.txt",
    "project_2_kalman_filter\Data\kalman_filter_data_velocity.txt"
]

fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': '3d'})

m = 0.027

for i, filename in enumerate(filenames):
    time, u, z = load_data(filename)
    sig, R, H = get_parameters(filename)
    del_t = get_del_t(time)
    if filename == "project_2_kalman_filter\Data\kalman_filter_data_velocity.txt":
        x_hat = np.zeros((6,1))
        x_hat[3:6,:] = z[0].reshape(3,1)
    else:
        x_hat = np.zeros((6,1))
        x_hat[:3,:] = z[0].reshape(3,1)
    # print(x_hat)
    estimates = kalman_filter(x_hat, np.diag([500] * 6), u, z, del_t, m, sig,time, R,H)
    ax = axs[i // 2, i % 2]
    ax.plot3D(estimates[:, 0], estimates[:, 1], estimates[:, 2], label='Estimated Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(plot_titles[i])

plt.tight_layout()
plt.show()
