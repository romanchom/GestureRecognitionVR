import numpy as np

def augument(data):
    # data is an array T x F (time, features)
    # T is irrelevant
    # F = 18, 3x4 matrix, 3 vector velocity, 3 vector angular velocity

    scale = np.identity(3, dtype='float32')
    scale *= np.random.lognormal(sigma=0.1, size=(3, 3)) * np.random.lognormal(sigma = 0.3)
    
    transform = np.identity(3, dtype='float32')
    transform += np.random.normal(scale=0.2, size=(3,3)) # random rotation/shear
    transform = np.dot(scale, transform) # random scale
    
    offset = np.random.normal(size=3)
    
    ret = np.empty_like(data)

    for i in range(data.shape[0]):
        vel = data[i, 12:15]
        ang_vel = data[i, 15:18]
        pos = data[i, [3, 7, 11]]
        matrix = data[i, [0, 1, 2, 4, 5, 6, 8, 9, 10]]
        matrix = np.reshape(matrix, (3, 3))

        vel = np.dot(transform, vel)
        ang_vel = np.dot(scale, vel)
        pos = np.dot(transform, pos) + offset
        matrix = np.dot(transform, matrix)

        ret[i, 12:15] = vel
        ret[i, 15:18] = ang_vel
        ret[i, [3, 7, 11]] = pos
        ret[i, [0, 1, 2, 4, 5, 6, 8, 9, 10]] = np.reshape(matrix, 9)

    return ret