import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from math import *

MU_0 = 4 * pi * (10**-7)

class WireSegment:
    def __init__(self, start, end, current):
        self.start = start
        self.end = end
        self.current = current

    def get_length(self):
        return (np.linalg.norm(self.end - self.start))
    
    def get_direction_vec(self):
        return ((self.end - self.start) / self.get_length())
    
    def get_B_at_point(self, point, iterations=500):
        """Calculates magnetic field vector at a point due to the wire segment
        Uses the Biot-Savart law with numeric integration for ease of implementation/avoiding doing the maths"""
        # r = point
        # B = [0, 0, 0]
        # for i in range(0, iterations):
        #     dl = (self.end - self.start) / iterations
        #     p = self.start + dl * i
        #     r_prime = r - p
        #     dB = (MU_0 / (4 * pi)) * self.current * np.cross(dl, r_prime) / (np.linalg.norm(r_prime) ** 3)
        #     B += dB

        # return B

        r_vec = point # point of interest
        x_0 = self.start + (np.dot((r_vec - self.start), self.get_direction_vec()) * self.get_direction_vec()) # closest point on line of wire to the point of interest, obtained using vector projections
        a = np.dot((x_0 - self.start), self.get_direction_vec())
        b = np.dot((x_0 - self.end), self.get_direction_vec())

        R = np.linalg.norm(r_vec - x_0) # minimum distance from the point of interest to wire

        # evaluation of integral
        B_norm = (MU_0 * self.current / (4 * pi * R)) * ((b / sqrt(b**2 + R**2)) - (a / sqrt(a**2 + R**2)))

        B_dir = np.cross(self.get_direction_vec(), r_vec - x_0)
        B_dir = B_dir / np.linalg.norm(B_dir)

        return B_dir * B_norm
    
class SquareHelmholtzCoil:
    def __init__(self, current, turns, side_length, centre, spacing, centre_axis, alignment, coil_width=0):
        """
        All lengths / coordinates in metres
        current - electrical current flowing through each coil (amps)
        turns - number of turns of wire in each coil
        centre - 3d coordinates of centre (midpoint between coils)
        side_length - Square side length of coils
        spacing - distance between coil centres. If the coils have a width, then this distance is measured from centre
                    to centre rather than inside edge to inside edge
        centre_axis - a 3d vector orthogonal to the coils. The sense defines the current direction.
                        If you point the thumb of your right hand in the direction of this vector, the direction your fingers
                        wrap is the current direction in the coils
        alignment - a 3d vector aligned with the direction of one of the coil sides
        coil_width - defaults to 0 for a perfectly thin coil. Can be set to a positive quantity to represent a coil
                with finite width from the first to the last turn of wire
        """
        self.current = current
        self.turns = turns
        self.a = side_length
        self.o = centre
        self.spacing = spacing
        self.z_axis = centre_axis
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)
        self.x_axis = np.cross(self.z_axis, alignment)
        self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)
        self.y_axis = np.cross(self.z_axis, self.x_axis)
        self.width = coil_width

        if (self.width == 0):
            # This is equivalent and makes computation faster, especially for a large number of turns
            self.current *= self.turns
            self.turns = 1
        
        self.wire_segments = []
        self.calculate_wires()

    def calculate_wires(self):
        u_x, u_y, u_z = self.z_axis
        # matrix representing 90 degree rotation about the coil's z axis (not the z axis of the global reference frame)
        rotation = np.array([[u_x**2,            u_x * u_y - u_z,    u_x * u_z + u_y ],
                             [u_x * u_y + u_z,   u_y**2,             u_y * u_z - u_x ],
                             [u_x * u_z - u_y,   u_y * u_z + u_x,    u_z**2          ]])
        
        start_corner_offset = ( - (self.x_axis * self.a / 2)
                                - (self.y_axis * self.a / 2))
        end_corner_offset   = (   (self.x_axis * self.a / 2)
                                - (self.y_axis * self.a / 2))
        
        for side in range(0, 4):
            if self.turns > 1:
                axial_offset = (-self.width/2) * self.z_axis
                turn_gap = (self.width/(self.turns - 1)) * self.z_axis
            else:
                axial_offset = 0
                turn_gap = 0
            for turn in range(0, self.turns):
                self.wire_segments.append(WireSegment(
                    self.o + start_corner_offset + axial_offset + (self.z_axis * self.spacing / 2),
                    self.o + end_corner_offset + axial_offset + (self.z_axis * self.spacing / 2),
                    self.current
                ))
                self.wire_segments.append(WireSegment(
                    self.o + start_corner_offset + axial_offset - (self.z_axis * self.spacing / 2),
                    self.o + end_corner_offset + axial_offset - (self.z_axis * self.spacing / 2),
                    self.current
                ))
                axial_offset += turn_gap
                

            start_corner_offset = rotation @ start_corner_offset
            end_corner_offset = rotation @ end_corner_offset

    def get_B_at_point(self, point):
        """Returns the magnetic field vector at a given point due to all the wires in the coil (tesla)"""
        B = 0
        for wire in self.wire_segments:
            B += wire.get_B_at_point(point)

        return B
    
coil_current = 0.75
coil_turns = 120
coil_side_length = 1
coil_spacing = 0.544506 * 1
coil_width = 0


hh_coil = SquareHelmholtzCoil(coil_current, coil_turns, coil_side_length, np.array([0, 0, 0]), coil_spacing, np.array([1, 0, 0]), np.array([0, 1, 0]), 0)

roi = 0.09 # region of interest (defines the side length of a cube centred on the origin.
            # All plots and stats are limited to this range)
quiver_resolution = coil_spacing / 6 # number of points to plot
# surface_resolution = 11 # number of points to plot on each axis over the roi for surface plots
# line_resolution = 51 # number of points to plot within the roi for line graphs

fig = plt.figure()

################ QUIVER AND COILS ####################
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_aspect('equal')
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")

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

    ax1.plot3D(wire_x, wire_y, wire_z, '#B87333')

# ax1.plot3D([-0.5, 0.5], [0, 0], [0, 0], 'black')

x_coords = []
y_coords = []
z_coords = []

B_x_values = []
B_y_values = []
B_Z_values = []

B_0 = hh_coil.get_B_at_point([0, 0, 0])
B_0_norm = np.linalg.norm(B_0)
B_max = B_0_norm
B_min = B_0_norm
deviation_max = 0
B_min_coords = (0, 0, 0)
B_max_coords = (0, 0, 0)
deviation_max_coords = (0, 0, 0)

for x in np.arange(-coil_spacing/2 + quiver_resolution, coil_spacing/2, quiver_resolution):
    for y in np.arange(-coil_side_length/2 + quiver_resolution, coil_side_length/2, quiver_resolution):
        for z in np.arange(-coil_side_length/2 + quiver_resolution, coil_side_length/2, quiver_resolution):
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

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

            B_x, B_y, B_z = B
            B_x_values.append(B_x)
            B_y_values.append(B_y)
            B_Z_values.append(B_z)

ax1.quiver(x_coords, y_coords, z_coords, B_x_values, B_y_values, B_Z_values, normalize = True, length = quiver_resolution/3)

ax1.scatter(B_min_coords[0], B_min_coords[1], B_min_coords[2])
ax1.scatter(B_max_coords[0], B_max_coords[1], B_max_coords[2], c="r")
ax1.scatter(deviation_max_coords[0], deviation_max_coords[1], [deviation_max_coords[2]], c="g")

print(f"Flux density at (0, 0, 0): {B_0_norm:.2e} T")
print(f"Max flux density: {B_max:.2e} T ({(100 * (B_max - B_0_norm) / B_0_norm):+.2f}%)")
print(f"Min flux density: {B_min:.2e} T ({(100 * (B_min - B_0_norm) / B_0_norm):+.2f}%)")
print(f"Max deviation: {deviation_max * 180 / pi:.2f} deg")

############## SURFACE PLOT ################
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

x_coords = np.linspace(-0.06, 0.06, 30)
y_coords = np.linspace(-0.06, 0.06, 30)
x_coords, y_coords = np.meshgrid(x_coords, y_coords)
field_strengths = []

for xl, yl in zip(x_coords, y_coords):
    field_strengths_l = []
    for x, y in zip(xl, yl):
        field_strengths_l.append(np.linalg.norm(hh_coil.get_B_at_point([x, y, 0])))
    field_strengths.append(field_strengths_l)


ax2.plot_surface(x_coords, y_coords, np.array(field_strengths))

# axbox1 = fig.add_axes([0.3, 0.05, 0.6, 0.05])
# side_length_tb = TextBox(axbox1, "Side length (m)", textalignment="left")
# axbox2 = fig.add_axes([0.3, 0.12, 0.6, 0.05])
# turns_tb = TextBox(axbox2, "Number of turns", textalignment="left")
# axbox3 = fig.add_axes([0.3, 0.19, 0.6, 0.05])
# current_tb = TextBox(axbox3, "Current (A)", textalignment="left")
# axbox4 = fig.add_axes([0.3, 0.26, 0.6, 0.05])
# spacing_tb = TextBox(axbox4, "Coil Spacing (m)", textalignment="left")

plt.show()
