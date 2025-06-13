# llpvtools: utilities for translating Landlab grids and fields
# into pyVista data structures for plotting.

import numpy as np
import pyvista as pv


def grid_to_pv(grid):
    """
    Create a pyVista DataSet from a Landlab grid and associated fields.
    """
    pass # this should be the 'master' function that hands off to other fns depending on grid type, etc.

# also, there should be at least two meshes: one for nodes, and one for corners

def get_reshaped_xyz(grid, field_for_z, at="node"):
    """
    Return x, y, and z coordinates of nodes or corners reshaped as
    (number of rows, number of columns). Use specified field for z.

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to be reshaped
    field_for_z : str
        Name of the field containing the z coordinate values
    at : str (optional)
        Name of grid elements: "node" (default) or "corner"

    Returns
    -------
    ndarray, ndarray, ndarray : x, y, and z coordinates reshaped

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((3, 4), 10.0)
    >>> z = grid.add_field("z", np.arange(12))
    >>> xx, yy, zz = get_reshaped_xyz(grid, "z")
    >>> xx
    array([[ 0., 10., 20., 30.],
           [ 0., 10., 20., 30.],
           [ 0., 10., 20., 30.]])
    >>> yy
    array([[ 0.,  0.,  0.,  0.],
           [10., 10., 10., 10.],
           [20., 20., 20., 20.]])
    >>> zz
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    """
    if at=="node":
        nr = grid.number_of_node_rows
        nc = grid.number_of_node_columns
        x = grid.x_of_node.reshape((nr, nc))
        y = grid.y_of_node.reshape((nr, nc))
        z = grid.at_node[field_for_z].reshape((nr, nc))
    elif at=="corner":
        nr = grid.number_of_corner_rows
        nc = grid.number_of_corner_columns
        x = grid.x_of_corner.reshape((nr, nc))
        y = grid.y_of_corner.reshape((nr, nc))
        z = grid.at_corner[field_for_z].reshape((nr, nc))
    else:
        raise(ValueError, "'at' must be 'node' or 'corner'")

    return x, y, z

def add_fields_to_pv_dataset(grid, dataset, z_field="", at="node", base_vals=0.0):
    """
    Add grid fields to a PyVista DataSet as data arrays.

    Parameters
    ----------
    grid : Landlab Grid object
        Grid containing fields to map
    dataset : PyVista DataSet object or subclass
        DataSet to which fields will be added
    z_field : str (optional)
        Name of the field that's used for the z-coordinate (default "")
    at : str (optional)
        Name of Landlab grid element (default "node")
    base_vals : float (optional)
        If 3d, value to use for bottom points (default 0.0)

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from pyvista import StructuredGrid
    >>> rmg = RasterModelGrid((3, 3))
    >>> z = rmg.add_field("z", np.arange(9))
    >>> _ = rmg.add_zeros("node_field1", at="node")
    >>> _ = rmg.add_zeros("node_field2", at="node")
    >>> pvsg = StructuredGrid(rmg.x_of_node.reshape((3, 3)), rmg.y_of_node.reshape((3, 3)), z.reshape((3, 3)))
    >>> _ = add_fields_to_pv_dataset(rmg, pvsg, z_field="z")
    >>> names = pvsg.array_names.copy()
    >>> names.sort()
    >>> names
    ['node_field1', 'node_field2']
    """
    is3d = False
    if at=="node":
        npts = grid.number_of_nodes
    else:
        npts = grid.number_of_corners
    if dataset.n_points==(2 * npts):
        is3d = True
    for field in grid.fields(include="at_" + at + "*"):
        fieldname = field[field.find(":") + 1:]
        if fieldname != z_field:
            vals = grid.field_values(fieldname, at=at)
            if is3d:
                vals = np.hstack((vals, base_vals + np.zeros(npts)))
            dataset.point_data[fieldname] = vals

def raster_grid_to_pv2d_struct(grid, field_for_z, at="node"):
    """
    Translate a RasterModelGrid into a PyVista 2D StructuredGrid.
    Includes node or corner fields as data arrays.

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to translate
    field_for_z : str
        Name of field to use for z coordinate
    at : str (optional)
        Which points to use: "node" (default) or "corner"

    Returns
    -------
    A PyVista StructuredGrid object

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((4, 5), 10.0)
    >>> z = grid.add_field("z", np.arange(grid.number_of_nodes), at="node")
    >>> struc_grd = raster_grid_to_pv2d_struct(grid, "z")
    >>> struc_grd.dimensions
    (4, 5, 1)

    >>> zc = grid.add_field("zc", np.arange(grid.number_of_corners), at="corner")
    >>> struc_grd = raster_grid_to_pv2d_struct(grid, "zc", at="corner")
    >>> struc_grd.dimensions
    (3, 4, 1)
    """
    x, y, z = get_reshaped_xyz(grid, field_for_z, at)
    struc_pv_grid = pv.StructuredGrid(x, y, z)
    add_fields_to_pv_dataset(grid, struc_pv_grid, at=at)
    return struc_pv_grid

def raster_grid_to_pv3d_struct(grid, field_for_z, at="node", depth=None):
    """
    Create and return a pyVista 3D StructuredGrid. Same as 
    raster_grid_to_pv2d_struct except that the PyVista grid
    is 3D, with two layers: one the z-coordinate of the Landlab
    grid, and the other a flat base at either a user-specified
    depth or a default value.

    ... TODO enable field for base...

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to translate
    field_for_z : str
        Name of field to use for z coordinate
    at : str (optional)
        Which points to use: "node" (default) or "corner"

    Returns
    -------
    PyVista StructuredGrid object of dimensions (nr, nc, 2),
    where nr and nc are the number of node (or corner) rows
    and columns, respectively, in the Landlab grid.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((4, 5), 10.0)
    >>> z = grid.add_field("z", np.arange(grid.number_of_nodes), at="node", depth=20.0)
    >>> raster_grid_to_pv3d_struct(grid, "z")
    StructuredGrid (...)
      N Cells:      12
      N Points:     40
      X Bounds:     0.000e+00, 4.000e+01
      Y Bounds:     0.000e+00, 3.000e+01
      Z Bounds:     -2.000e+01, 1.900e+01
      Dimensions:   4, 5, 2
      N Arrays:     1

    >>> zc = grid.add_field("zc", np.arange(grid.number_of_corners), at="corner")
    >>> raster_grid_to_pv3d_struct(grid, "zc", at="corner")
    StructuredGrid (...)
      N Cells:      6
      N Points:     24
      X Bounds:     5.000e+00, 3.500e+01
      Y Bounds:     5.000e+00, 2.500e+01
      Z Bounds:     -1.500e+01, 1.100e+01
      Dimensions:   3, 4, 2
      N Arrays:     1
    """

    if at=="node":
        x = grid.x_of_node
        y = grid.y_of_node
        z = grid.at_node[field_for_z]
        nr = grid.number_of_node_rows
        nc = grid.number_of_node_columns
    elif at=="corner":
        x = grid.x_of_corner
        y = grid.y_of_corner
        z = grid.at_corner[field_for_z]
        nr = grid.number_of_corner_rows
        nc = grid.number_of_corner_columns
    else:
        raise(ValueError, "'at' must be 'node' or 'corner'")
    
    # default bottom of grid is flat surface at depth below lowest point
    # equal to half the widest grid extent
    if depth is None:
        depth = max(np.amax(x) - np.amin(x),
                    np.amax(y) - np.amin(y)) / 2
        depth -= np.amin(z)

    top = np.column_stack((x, y, z))
    bottom = top.copy()
    bottom[:,2] = -depth

    vol = pv.StructuredGrid()
    vol.points = np.vstack((top, bottom))
    vol.dimensions = [nr, nc, 2]

    add_fields_to_pv_dataset(grid, vol, at=at)  
    
    return vol
