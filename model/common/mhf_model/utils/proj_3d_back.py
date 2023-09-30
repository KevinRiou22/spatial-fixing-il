def prj_3dto2d(3d_cord, K, R, t):
    P = K @ np.hstack([R, t])
    print(P.shape)  # (3, 4)
    X_homo = np.hstack([3d_cord, 1.0])  # convert to homogenous
    x_homo = P @ X_homo  # project
    x = x_homo[:2] / x_homo[2]  # convert back to cartesian (check that x_homo[2] > 0)
    return x