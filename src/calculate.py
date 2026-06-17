import hhcoil as hhc
import hhplot as hhp

import numpy as np
from matplotlib import pyplot as plt
from enum import Enum

IDEAL_SQUARE_COIL_SEPARATION = 0.544506
class ROIShape(Enum):
    CUBE = 0
    SPHERE = 1

# A simple matplotlib based calculator for square Helmholtz coils. Change the 
# parameters below and run the program to see the results.

########## CHANGE THESE PARAMETERS ##########

COIL_CURRENT = 1 # A
COIL_TURNS = 60
COIL_SIDE_LENGTH = 0.65 # m
COIL_SPACING = COIL_SIDE_LENGTH * IDEAL_SQUARE_COIL_SEPARATION # m
COIL_WIDTH = 0 # m. Ideal coils are considered to have 0 width. To see the effect
                # of a finite width, change this value. However a non-zero width
                # may make computation times excessive if the number of turns is high.
ROI_SHAPE = ROIShape.SPHERE
ROI_CUBE_SIDE_LENGTH = 0.18 # metres. Defines the side length of a cube centred on the origin.
            # All plots and stats are limited to this range)
ROI_SPHERE_DIAMETER = 0.18 # metres. Defines the diameter of a sphere centred on the origin.
            # All plots and stats are limited to this range)

#############################################


hh_coil = hhc.SquareHelmholtzCoil(COIL_CURRENT, COIL_TURNS, COIL_SIDE_LENGTH, COIL_SPACING, COIL_WIDTH,
                                   np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

fig = plt.figure("Helmholtz Coil Calculator", layout="constrained")
axes = fig.subplot_mosaic(
    [
        ["coils", "xy", "x"],
        ["coils", "yz", "y"]
    ],
    per_subplot_kw={
        ("coils", "xy", "yz"): {"projection": "3d"},
    },
)

# plt.subplots_adjust(wspace=0.5, hspace=0.5)

print("Calculating region of interest parameters...")
match ROI_SHAPE:
    case ROIShape.CUBE:
        hhp.print_roi_cube_data(hh_coil, ROI_CUBE_SIDE_LENGTH, samples=35)
    case ROIShape.SPHERE:
        hhp.print_spherical_roi_data(hh_coil, ROI_SPHERE_DIAMETER, samples=35)
    case _:
        print("Unrecognised region of interest shape.")

print("Constructing plots...")
hhp.plot_coils(axes["coils"], hh_coil)
if (ROI_SHAPE == ROIShape.CUBE):
    hhp.plot_roi_cube(axes["coils"], ROI_CUBE_SIDE_LENGTH)

plot_range = ROI_CUBE_SIDE_LENGTH if ROI_SHAPE == ROIShape.CUBE else ROI_SPHERE_DIAMETER
hhp.plot_field_over_plane(axes["xy"], "xy", plot_range, hh_coil)
hhp.plot_field_over_plane(axes["yz"], "yz", plot_range, hh_coil)

hhp.plot_field_along_axis(axes["x"], "x", plot_range, hh_coil)
hhp.plot_field_along_axis(axes["y"], "y", plot_range, hh_coil)

print("Complete.")

plt.show()

