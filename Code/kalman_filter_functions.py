import numpy as np

def get_parameters(filename):
    if filename == "project_2_kalman_filter\Data\kalman_filter_data_velocity.txt":
        sig = 0.01
        R = np.eye(3) * (0.5)**2

        H = np.array([0,0,0,1,0,0,
                    0,0,0,0,1,0,
                    0,0,0,0,0,1]).reshape(3,6)

    elif filename == "project_2_kalman_filter\Data\kalman_filter_data_low_noise.txt":
        sig = 0.01
        R = np.eye(3) * (0.5)**2

        H = np.array([1,0,0,0,0,0,
                    0,1,0,0,0,0,
                    0,0,1,0,0,0]).reshape(3,6)
        
    elif filename == "project_2_kalman_filter\Data\kalman_filter_data_high_noise.txt":
        sig = 0.01
        R = np.eye(3) * (1.0)**2

        H = np.array([1,0,0,0,0,0,
                    0,1,0,0,0,0,
                    0,0,1,0,0,0]).reshape(3,6)
        
    else:
        sig = 0.001
        R = np.eye(3) * (0.01)**2
        H = np.array([1,0,0,0,0,0,
                    0,1,0,0,0,0,
                    0,0,1,0,0,0]).reshape(3,6)
    return sig, R, H

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    time = data[:, 0]
    u = data[:, 1:4]
    z = data[:, 4:7]
    return time, u, z

def get_del_t(time):
    del_t = np.mean(np.diff(time))
    return del_t

def matrix_maker(del_t, m, sig):
    F = np.eye(6)
    F[0, 3] = del_t
    F[1, 4] = del_t
    F[2, 5] = del_t

    G = np.zeros((6, 3))
    G[3:6, :] = np.eye(3) * del_t / m

    Q = sig**2 * (G @ G.T)
    return F, G, Q

def predict_step(u, x_hat, P, F, G, Q):
    u = u.reshape(3, 1)
    x_hat = (F @ x_hat) + (G @ u)
    P = F @ (P @ F.T) + Q
    return x_hat, P

def update_step(z, x_hat, P, R,H): 
    K = P @ (H.T @ np.linalg.inv(H @ (P @ H.T) + R))
    z = z.reshape(3, 1)
    x_hat = x_hat + K @ (z - H @ x_hat)
    P = (np.eye(6) - K @ H) @ P @ (np.eye(6) - K @ H).T + K @ R @ K.T
    return x_hat, P

def kalman_filter(x_hat, P, u, z, del_t, m, sig, time, R, H):
    F, G, Q = matrix_maker(del_t, m, sig)
    estimates = []

    for i in range(len(time)):
        x_hat, P = predict_step(u[i], x_hat, P, F, G, Q)
        x_hat, P = update_step(z[i], x_hat, P, R,H)
        estimates.append(x_hat.flatten())
    return np.array(estimates)