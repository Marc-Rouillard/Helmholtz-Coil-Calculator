import numpy as np
from math import pi, sqrt

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
        """Calculate start and end points of each wire segment"""
        self.wire_segments = []
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
