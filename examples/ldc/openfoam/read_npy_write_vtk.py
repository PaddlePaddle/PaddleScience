import numpy as np
from pyevtk.hl import pointsToVTK

fname = "ldc2d_unsteady_Re100"
ldc2d_u = np.load('openfoam_u_100.npy')
print(ldc2d_u.shape)
np.savetxt('openfoam_u_100.txt', ldc2d_u, delimiter=" ")
print('ldc2d_u.shape: ', ldc2d_u.shape)

ldc2d_v = np.load('openfoam_v_100.npy')
np.savetxt('openfoam_v_100.txt', ldc2d_v, delimiter=" ")
print('ldc2d_v.shape: ', ldc2d_v.shape)

# fname = "ldc2d_unsteady_Re400"
# ldc2d_u = np.load('openfoam_u_400.npy')
# np.savetxt('openfoam_u_400.txt', ldc2d_u, delimiter=" ")
# print(ldc2d_u.shape)

# ldc2d_v = np.load('openfoam_v_400.npy')
# np.savetxt('openfoam_v_400.txt', ldc2d_v, delimiter=" ")
# print(ldc2d_v.shape)

data = np.stack((ldc2d_u, ldc2d_v), axis=1)
print("data.shape: ", data.shape)

xx = np.linspace(start=-0.5, stop=0.5, num=101)
yy = np.linspace(start=-0.5, stop=0.5, num=101)
ldc2d_x, ldc2d_y = np.meshgrid(xx, yy)
print('ldc2d_x.shape: ', ldc2d_x.shape)
print('ldc2d_y.shape: ', ldc2d_y.shape)
cords = np.stack((ldc2d_x.flatten(), ldc2d_y.flatten()), axis=1)
print('cords.shape: ', cords.shape)

num_cords = data.shape[0]
axis_z = np.zeros(num_cords, dtype="float32")
pointsToVTK(
    fname,
    cords[:, 0].copy().astype("float32"),
    cords[:, 1].copy().astype("float32"),
    axis_z,
    data={
        "u": data[:, 0].copy(),
        "v": data[:, 1].copy()
    }
)
print("Successfully saved results: ", fname)