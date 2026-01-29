import hhcoil as hhc
import hhplot as hhp

import numpy as np
from matplotlib import pyplot as plt

########## CHANGE THESE PARAMETERS ##########

coil_current = 0.75
coil_turns = 125
coil_side_length = 1
coil_spacing = 0.6
coil_width = 0 # Ideal coils are considered to have 0 width. To see the effect
                # of a finite width, change this value. However a non-zero width
                # may make computation times excessive if the number of turns is high.
roi = 0.3 # region of interest (defines the side length of a cube centred on the origin.
            # All plots and stats are limited to this range)

#############################################


hh_coil = hhc.SquareHelmholtzCoil(coil_current, coil_turns, coil_side_length, coil_spacing, coil_width,
                                   np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0]))

fig = plt.figure(layout="constrained")
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

hhp.plot_coils(axes["coils"], hh_coil)
hhp.plot_roi(axes["coils"], roi)

hhp.print_roi_data(hh_coil, roi)

hhp.plot_field_over_plane(axes["xy"], "xy", roi, hh_coil)
hhp.plot_field_over_plane(axes["yz"], "yz", roi, hh_coil)

hhp.plot_field_over_axis(axes["x"], "x", roi, hh_coil)
hhp.plot_field_over_axis(axes["y"], "y", roi, hh_coil)

plt.show()

