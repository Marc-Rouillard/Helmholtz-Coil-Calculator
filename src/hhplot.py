import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
from math import acos, pi

from hhcoil import SquareHelmholtzCoil

DEFAULT_QUIVER_SAMPLES = 5
DEFAULT_SURFACE_SAMPLES = 25
DEFAULT_LINE_SAMPLES = 45
DEFAULT_CUBE_ROI_SAMPLES = 15
DEFAULT_SPHERICAL_ROI_SAMPLES = 25;

def plot_coils(ax :matplotlib.axes.Axes, hh_coil: SquareHelmholtzCoil,
               quivers: bool=True, quiver_samples: int=DEFAULT_QUIVER_SAMPLES):
    """
    Plot coils on 3d axes. Optionally also plot quivers (vector arrows) showing
    field direction.
    
    :param ax: matplotlib.axes.Axes object on which to plot (3d projection)
    :param hh_coil: SquareHelmholtzCoil object
    :param quivers: `True` if vector field should be plotted (default `True`)
    :param quiver_samples: number of quivers to plot along each axis
    """
    ax.set_title("Coils")
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    coil_spacing = hh_coil.spacing
    coil_side_length = hh_coil.a

    # invisible bounding cube to make the axis scales equal
    bb_side_length = max(coil_spacing, coil_side_length)
    for i in (-1, 1):
        for j in (-1, 1):
            for k in (-1, 1):
                ax.plot(i * bb_side_length/2, j * bb_side_length/2, k * bb_side_length/2)

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

        ax.plot3D(wire_x, wire_y, wire_z, '#B87333')

        if quivers:
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

            ax.quiver(x_coords, y_coords, z_coords,
                      B_x_values, B_y_values, B_Z_values,
                      normalize = True,
                      length = 0.3 * bb_side_length/(quiver_samples - 1))
            
def plot_roi_cube(ax: matplotlib.axes.Axes, roi_range: float):
    """
    Plot a wire frame cube to show the region of interest
    
    :param ax: matplotlib.axes.Axes object on which to plot
    :param roi_range: size of the region of interest (cube side length)
    """
    for i in (-roi_range/2, roi_range/2):
        for j in (-roi_range/2, roi_range/2):
            ax.plot([-roi_range/2, roi_range/2], [i, i], [j, j], c="black")
            ax.plot([i, i], [-roi_range/2, roi_range/2], [j, j], c="black")
            ax.plot([i, i], [j, j], [-roi_range/2, roi_range/2], c="black")

def print_roi_cube_data(hh_coil: SquareHelmholtzCoil, roi_range: float,
                           ax: matplotlib.axes.Axes=None,
                           samples: int = DEFAULT_CUBE_ROI_SAMPLES):
    """
    Find maximum field strength, minimum field strength and maximum angular
    deviation within region of interest and optionally plot the locations of each point.
    This function works by just sampling points within the region of interest.
    It is a crude method, but will still be quite accurate when the region of
    interest is small with respect to the coils. However, it may be less
    accurate with a larger roi or very low sample count.
    
    :param hh_coil: SquareHelmholtzCoil object
    :param roi_range: range to check, centred on origin (defines side length of 
                roi cube)
    :param axes: if not None, the locations of each point of interest will be
                plotted on these axes.
    :param samples: number of samples to take along each axis
    """
    B_0 = hh_coil.get_B_at_point([0, 0, 0])
    B_0_norm = np.linalg.norm(B_0)
    B_max = B_0_norm
    B_min = B_0_norm
    deviation_max = 0
    B_min_coords = (0, 0, 0)
    B_max_coords = (0, 0, 0)
    deviation_max_coords = (0, 0, 0)

    for x in np.linspace(-roi_range/2, roi_range/2, samples):
        for y in np.linspace(-roi_range/2, roi_range/2, samples):
            for z in np.linspace(-roi_range/2, roi_range/2, samples):
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

    # Print results
    print(f"Over {int(roi_range * 1000)}x{int(roi_range * 1000)}x{int(roi_range * 1000)} mm volume:")
    print(f"    Flux density at (0, 0, 0): {B_0_norm:.3e} T")
    print(f"    Max flux density: {B_max:.3e} T ({(100 * (B_max - B_0_norm) / B_0_norm):+.3f}%)")
    print(f"    Min flux density: {B_min:.3e} T ({(100 * (B_min - B_0_norm) / B_0_norm):+.3f}%)")
    print(f"    Max deviation: {deviation_max * 180 / pi:.3f} deg")

    # Plot locations
    if ax is not None:
        ax.scatter(B_min_coords[0], B_min_coords[1], B_min_coords[2])
        ax.scatter(B_max_coords[0], B_max_coords[1], B_max_coords[2], c="r")
        ax.scatter(deviation_max_coords[0], deviation_max_coords[1],
                            [deviation_max_coords[2]], c="g")
        
def print_spherical_roi_data(hh_coil: SquareHelmholtzCoil, roi_diameter: float,
                           ax: matplotlib.axes.Axes=None,
                           samples: int = DEFAULT_SPHERICAL_ROI_SAMPLES):
    """
    Find maximum field strength, minimum field strength and maximum angular
    deviation within region of interest and optionally plot the locations of each point.
    This function works by just sampling points within the region of interest.
    It is a crude method, but will still be quite accurate when the region of
    interest is small with respect to the coils. However, it may be less
    accurate with a larger roi or very low sample count.
    
    :param hh_coil: SquareHelmholtzCoil object
    :param roi_range: range to check, centred on origin (defines side length of 
                roi cube)
    :param axes: if not None, the locations of each point of interest will be
                plotted on these axes.
    :param samples: number of samples to take along each axis
    """
    B_0 = hh_coil.get_B_at_point([0, 0, 0])
    B_0_norm = np.linalg.norm(B_0)
    B_max = B_0_norm
    B_min = B_0_norm
    deviation_max = 0
    B_min_coords = (0, 0, 0)
    B_max_coords = (0, 0, 0)
    deviation_max_coords = (0, 0, 0)

    for x in np.linspace(-roi_diameter/2, roi_diameter/2, samples):
        for y in np.linspace(-roi_diameter/2, roi_diameter/2, samples):
            for z in np.linspace(-roi_diameter/2, roi_diameter/2, samples):
                if (x**2 + y**2 + z**2 < (roi_diameter**2)/4):
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

    # Print results
    print(f"Over {int(roi_diameter * 1000)} mm diameter sphere:")
    print(f"    Flux density at (0, 0, 0): {B_0_norm:.3e} T")
    print(f"    Max flux density: {B_max:.3e} T ({(100 * (B_max - B_0_norm) / B_0_norm):+.3f}%)")
    print(f"    Min flux density: {B_min:.3e} T ({(100 * (B_min - B_0_norm) / B_0_norm):+.3f}%)")
    print(f"    Max deviation: {deviation_max * 180 / pi:.3f} deg")

    # Plot locations
    if ax is not None:
        ax.scatter(B_min_coords[0], B_min_coords[1], B_min_coords[2])
        ax.scatter(B_max_coords[0], B_max_coords[1], B_max_coords[2], c="r")
        ax.scatter(deviation_max_coords[0], deviation_max_coords[1],
                            [deviation_max_coords[2]], c="g")

def plot_field_over_plane(ax: matplotlib.axes.Axes, plane: str, range: float,
                          hh_coil: SquareHelmholtzCoil,
                          samples: int=DEFAULT_SURFACE_SAMPLES):
    """
    Plot magnetic field over a plane as a surface on a set of 3d axes.
    
    :param ax: matplotlib.axes.Axes object on which to plot
    :param plane: can be 'xy', 'yz' or 'xz'
    :param range: range to plot, centred on origin.
                (i.e if range=3, the plot will go from -0.15 to 0.15)
    : param samples: number of samples to take along each axis (optional)
    """
    # 'a' and 'b' are used to refer to the graph x and y axes to avoid
    # confusion with the axes of the coil's coordinate system
    a_coords = []
    b_coords = []
    field_strengths = []

    for i, a in enumerate(np.linspace(-range/2, range/2, samples)):
        a_coords.append([])
        b_coords.append([])
        field_strengths.append([])
        for b in np.linspace(-range/2, range/2, samples):
            a_coords[i].append(a)
            b_coords[i].append(b)
            match plane:
                case "xy":
                    coords = np.array([a, b, 0])
                case "yz":
                    coords = np.array([0, a, b])
                case "xz":
                    coords = np.array([a, 0, b])
                case _:
                    raise ValueError("Parameter `plane` must be one of 'xy', "
                                        "'yz' or 'xz'")
            field_strengths[i].append(np.linalg.norm(hh_coil.get_B_at_point(coords)))

    ax.plot_surface(a_coords, b_coords, np.array(field_strengths))
    ax.set_title(f"Field intensity over {plane} plane")
    ax.set_xlabel(f"{plane[0]} (m)")
    ax.set_ylabel(f"{plane[1]} (m)")
    ax.set_zlabel("B (T)")
    ax.ticklabel_format(axis="z", style="sci", scilimits=(-2, 2))

def plot_field_along_axis(ax: matplotlib.axes.Axes, axis: str, range: float,
                         hh_coil: SquareHelmholtzCoil,
                         samples=DEFAULT_LINE_SAMPLES):
    """
    Plot magnetic field along an axis as a line graph.
    
    :param ax: matplotlib.axes.Axes object on which to plot
    :param axis: can be 'x', 'y' or 'z'
    :param range: range to plot, centred on origin.
        (i.e if range=0.3, the plot will go from -0.15 to 0.15)
    :param samples: number of samples to take along each axis (optional)
    """
    points = np.linspace(-range/2, range/2, samples)
    match axis:
        case "x":
            B_values = [np.linalg.norm(hh_coil.get_B_at_point([p, 0, 0])) for p in points]
        case "y":
            B_values = [np.linalg.norm(hh_coil.get_B_at_point([0, p, 0])) for p in points]
        case "z":
            B_values = [np.linalg.norm(hh_coil.get_B_at_point([0, 0, p])) for p in points]
        case _:
            raise ValueError("Parameter `axis` must be one of 'x', 'y' or 'z'.")

    ax.set_title(f"Field intensity along {axis} axis")
    ax.plot(points, B_values)
    ax.set_xlabel(f"{axis} (m)")
    ax.set_ylabel("B (T)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
