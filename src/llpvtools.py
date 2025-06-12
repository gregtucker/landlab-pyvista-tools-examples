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

def add_fields_to_pv_dataset(grid, dataset, z_field="", at="node"):
    """
    Add grid fields to a pyVista DataSet as data arrays.

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

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from pyvista import StructuredGrid
    >>> rmg = RasterModelGrid((3, 3))
    >>> z = rmg.add_ones("z", at="node")
    >>> _ = rmg.add_zeros("node_field1", at="node")
    >>> _ = rmg.add_zeros("node_field2", at="node")
    >>> pvsg = StructuredGrid(rmg.x_of_node.reshape((3, 3)), rmg.y_of_node.reshape((3, 3)), z.reshape((3, 3)))
    >>> _ = add_fields_to_pv_dataset(rmg, pvsg, z_field="z")
    >>> pvsg.array_names
    ['node_field2', 'node_field1']
    """
    for field in grid.fields(include="at_" + at + "*"):
        fieldname = field[field.find(":") + 1:]
        if fieldname != z_field:
            dataset.point_data[fieldname] = grid.field_values(fieldname, at=at)

def raster_grid_to_pv2d_struct(grid, field_for_z, at="node"):
    x, y, z = get_reshaped_xyz(grid, field_for_z, at)
    struc_pv_grid = pv.StructuredGrid(x, y, z)
    add_fields_to_pv_dataset(grid, struc_pv_grid, at=at)
    return struc_pv_grid

def raster_grid_to_pv3d_struct(grid, field_for_z, at="node", depth=None):
    """
    Create and return a pyVista 3D StructuredGrid.

    The 3D nature comes from having two "layers": "field_for_z" on top, and ... TODO enable field for base...
    """

    # default bottom of grid is flat surface at depth equal to half the widest grid extent
    if depth is None:
        depth = max(np.amax(grid.x_of_node) - np.amin(grid.x_of_node),
                    np.amax(grid.y_of_node) - np.amin(grid.y_of_node)) / 2

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
    
    top = np.column_stack((x, y, z))
    bottom = top.copy()
    bottom[:,2] =  -depth

    vol = pv.StructuredGrid()
    vol.points = np.vstack((top, bottom))
    vol.dimensions = [nr, nc, 2]

    print(vol.dimensions)
          
    return vol
