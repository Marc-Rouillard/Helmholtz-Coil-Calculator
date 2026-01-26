import numpy as np
import matplotlib.pyplot as plt
from math import acos, pi

import hhcoil as hh

coil_current = 0.75
coil_turns = 125
coil_side_length = 1
coil_spacing = 0.6
coil_width = 0


hh_coil = hh.SquareHelmholtzCoil(coil_current, coil_turns, coil_side_length, np.array([0, 0, 0]), coil_spacing, np.array([1, 0, 0]), np.array([0, 1, 0]), coil_width)

roi = 1.2 # region of interest (defines the side length of a cube centred on the origin.
            # All plots and stats are limited to this range)


quiver_samples = 7 # number of points to sample along each axis to plot quiver

surface_samples = 29 # number of points to plot on each axis over the roi for surface plots
line_samples = 49 # number of points to plot within the roi for line graphs

fig = plt.figure(layout="none")
axes = fig.subplot_mosaic(
    [
        ["coils", "xy", "x"],
        ["coils", "yz", "y"]
    ],
    per_subplot_kw={
        ("coils", "xy", "yz"): {"projection": "3d"},
    },
)



################ QUIVER AND COILS ####################
coil_ax = axes["coils"]
coil_ax.set_aspect('equal')
coil_ax.set_xlabel("x (m)")
coil_ax.set_ylabel("y (m)")
coil_ax.set_zlabel("z (m)")

# bounding cube to make the axis scales equal
bb_side_length = max(coil_spacing, coil_side_length)
for i in (-1, 1):
    for j in (-1, 1):
        for k in (-1, 1):
            coil_ax.plot(i * bb_side_length/2, j * bb_side_length/2, k * bb_side_length/2)

# roi cube
for i in (-roi/2, roi/2):
    for j in (-roi/2, roi/2):
        coil_ax.plot([-roi/2, roi/2], [i, i], [j, j], c="black")
        coil_ax.plot([i, i], [-roi/2, roi/2], [j, j], c="black")
        coil_ax.plot([i, i], [j, j], [-roi/2, roi/2], c="black")

# Plot coils
for wire in hh_coil.wire_segments:
    wire_x = []
    wire_y = []
    wire_z = []

    wire_x.append(wire.start[0])
    wire_y.append(wire.start[1])
    wire_z.append(wire.start[2])
    wire_x.append(wire.end[0])
    wire_y.append(wire.end[1])
    wire_z.append(wire.end[2])

    coil_ax.plot3D(wire_x, wire_y, wire_z, '#B87333')

# ax1.plot3D([-0.5, 0.5], [0, 0], [0, 0], 'black')

x_coords = []
y_coords = []
z_coords = []

B_x_values = []
B_y_values = []
B_Z_values = []

for x in np.linspace(-bb_side_length/2, bb_side_length/2, quiver_samples):
    for y in np.linspace(-bb_side_length/2, bb_side_length/2, quiver_samples):
        for z in np.linspace(-bb_side_length/2, bb_side_length/2, quiver_samples):
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

            B_x, B_y, B_z = hh_coil.get_B_at_point([x, y, z])
            B_x_values.append(B_x)
            B_y_values.append(B_y)
            B_Z_values.append(B_z)


coil_ax.quiver(x_coords, y_coords, z_coords, B_x_values, B_y_values, B_Z_values, normalize = True, length = 0.3 * bb_side_length/(quiver_samples - 1))

####### Examine roi to find minima and maxima ###########
roi_samples = 11 # number of samples to take along each axis within the roi to determine max/min field strength and maximum deviation
# This is a very simple method but produces pretty accurate results when the roi is smallish wrt the coils, especially since
# the min and max field strengths typically lie on the main axes at the edges of the roi so they will be sampled exactly

B_0 = hh_coil.get_B_at_point([0, 0, 0])
B_0_norm = np.linalg.norm(B_0)
B_max = B_0_norm
B_min = B_0_norm
deviation_max = 0
B_min_coords = (0, 0, 0)
B_max_coords = (0, 0, 0)
deviation_max_coords = (0, 0, 0)

for x in np.linspace(-roi/2, roi/2, roi_samples):
    for y in np.linspace(-roi/2, roi/2, roi_samples):
        for z in np.linspace(-roi/2, roi/2, roi_samples):
            B = hh_coil.get_B_at_point([x, y, z])
            B_norm = np.linalg.norm(B)
            if B_norm < B_min:
                B_min = B_norm
                B_min_coords = (x, y, z)
            elif B_norm > B_max:
                B_max = B_norm
                B_max_coords = (x, y, z)


            deviation = acos(np.dot(B, B_0) / (B_0_norm * np.linalg.norm(B)))
            if deviation > deviation_max:
                deviation_max = deviation
                deviation_max_coords = (x, y, z)

coil_ax.scatter(B_min_coords[0], B_min_coords[1], B_min_coords[2])
coil_ax.scatter(B_max_coords[0], B_max_coords[1], B_max_coords[2], c="r")
coil_ax.scatter(deviation_max_coords[0], deviation_max_coords[1], [deviation_max_coords[2]], c="g")

print(f"Flux density at (0, 0, 0): {B_0_norm:.2e} T")
print(f"Max flux density: {B_max:.2e} T ({(100 * (B_max - B_0_norm) / B_0_norm):+.2f}%)")
print(f"Min flux density: {B_min:.2e} T ({(100 * (B_min - B_0_norm) / B_0_norm):+.2f}%)")
print(f"Max deviation: {deviation_max * 180 / pi:.2f} deg")

############## SURFACE PLOT ################
xy_ax = axes["xy"]

x_coords = np.linspace(-roi/2, roi/2, surface_samples)
y_coords = np.linspace(-roi/2, roi/2, surface_samples)
x_coords, y_coords = np.meshgrid(x_coords, y_coords)
field_strengths = []

for xl, yl in zip(x_coords, y_coords):
    field_strengths_l = []
    for x, y in zip(xl, yl):
        field_strengths_l.append(np.linalg.norm(hh_coil.get_B_at_point([x, y, 0])))
    field_strengths.append(field_strengths_l)

xy_ax.plot_surface(x_coords, y_coords, np.array(field_strengths))
xy_ax.set_xlabel("x (m)")
xy_ax.set_ylabel("y (m)")
xy_ax.set_zlabel("B (T)")
xy_ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))

yz_ax = axes["yz"]

y_coords = np.linspace(-roi/2, roi/2, surface_samples)
z_coords = np.linspace(-roi/2, roi/2, surface_samples)
y_coords, z_coords = np.meshgrid(y_coords, z_coords)
field_strengths = []

for yl, zl in zip(y_coords, z_coords):
    field_strengths_l = []
    for y, z in zip(yl, zl):
        field_strengths_l.append(np.linalg.norm(hh_coil.get_B_at_point([0, y, z])))
    field_strengths.append(field_strengths_l)


yz_ax.plot_surface(y_coords, z_coords, np.array(field_strengths))
yz_ax.set_xlabel("y (m)")
yz_ax.set_ylabel("z (m)")
yz_ax.set_zlabel("B (T)")
yz_ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))

##### AXIS PLOTS ######
x_ax = axes["x"]
x_coords = np.linspace(-roi/2, roi/2, line_samples)
B_values = [np.linalg.norm(hh_coil.get_B_at_point([x, 0, 0])) for x in x_coords]
x_ax.plot(x_coords, B_values)
x_ax.set_xlabel("x (m)")
x_ax.set_ylabel("B (T)")
x_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

y_ax = axes["y"]
y_coords = np.linspace(-roi/2, roi/2, line_samples)
B_values = [np.linalg.norm(hh_coil.get_B_at_point([0, y, 0])) for y in y_coords]
y_ax.plot(y_coords, B_values)
y_ax.set_xlabel("y (m)")
y_ax.set_ylabel("B (T)")
y_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

plt.show()
