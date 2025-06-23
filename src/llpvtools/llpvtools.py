"""
llpvtools: utilities for translating Landlab grids and fields
into pyVista data structures for plotting.

Greg Tucker, University of Colorado Boulder, USA.
Original development: June 2025
"""

import numpy as np
import pyvista as pv
from landlab import RasterModelGrid, IcosphereGlobalGrid, NetworkModelGrid
from landlab.utils.decorators import use_field_name_array_or_value


def _has_dual(grid):
    """
    True if grid has dual complement (corners-faces-cells), False otherwise.

    Parameters
    ----------
    grid : Landlab grid object
        The grid to inspect.

    Returns
    -------
    bool : True if grid has corners (vertices of cells) defined

    Examples
    --------
    >>> from landlab import RasterModelGrid, NetworkModelGrid
    >>> _has_dual(RasterModelGrid((3, 3)))
    True
    >>> _has_dual(NetworkModelGrid(((0, 1), (0,0)), ((1, 0))))
    False
    """
    try:
        _ = grid.number_of_corners
        return True
    except AttributeError:
        return False


def grid_to_pv(
    grid,
    field_for_node_z=None,
    field_for_corner_z=None,
    make3d=False,
    values_for_node_base=None,
    values_for_corner_base=None,
):
    """
    Create pyVista DataSet(s) from a Landlab grid and associated fields.

    Most Landlab grids have dual meshes (nodes-links-patches and corners-faces-cells),
    and where that's the case, the function returns one PyVista mesh object for each.

    Parameters
    ----------
    grid : Landlab model grid
        Grid to be translated into PyVista mesh(es)
    field_for_node_z : str, ndarray, or float (optional)
        Values for node z coords (default "topographic__elevaton" if exists, or 0)
    field_for_corner_z : str, ndarray, or float (optional)
        Values for corner z coords (default "topographic__elevaton" if exists, or 0)
    make3d : optional
        Option to make a 3D mesh, with a top and bottom layer (default False)
    values_for_node_base : str, ndarray, or float (optional)
        Values for z of lower node layer (default 1/2 a grid width below lowest top z)
    values_for_corner_base : str, ndarray, or float (optional)
        Values for z of lower corner layer (default 1/2 a grid width below lowest top z)

    Returns
    -------
    PyVista.StructuredGrid, PyVista.StructuredGrid or None
        A StructuredGrid for nodes, one for corners (or None if no defined corners)

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> rmg = RasterModelGrid((3, 3))
    >>> pvm_nodes, pvm_corners = grid_to_pv(rmg)
    >>> type(pvm_nodes)
    <class 'pyvista.core.pointset.StructuredGrid'>
    >>> pvm_nodes.n_points
    9
    >>> pvm_corners.n_points
    4
    """
    if field_for_node_z is None:
        if "topographic__elevation" in grid.at_node.keys():
            field_for_node_z = "topographic__elevation"
        else:
            field_for_node_z = 0.0
    if _has_dual(grid):
        if field_for_corner_z is None:
            if "topographic__elevation" in grid.at_node.keys():
                field_for_corner_z = "topographic__elevation"
            else:
                field_for_corner_z = 0.0

    if isinstance(grid, RasterModelGrid):
        if make3d:
            pv_mesh_nodes = raster_grid_to_pv3d_struct(
                grid, field_for_node_z, at="node", base_vals=values_for_node_base
            )
            pv_mesh_corners = raster_grid_to_pv3d_struct(
                grid, field_for_corner_z, at="corner", base_vals=values_for_corner_base
            )
        else:
            pv_mesh_nodes = raster_grid_to_pv2d_struct(
                grid, field_for_node_z, at="node"
            )
            pv_mesh_corners = raster_grid_to_pv2d_struct(
                grid, field_for_corner_z, at="corner"
            )
    elif isinstance(grid, IcosphereGlobalGrid):
        pv_mesh_nodes = non_raster_grid_to_pv_unstructured(
            grid,
            field_or_array_for_z=grid.z_of_node,
            at="node",
            is3d=False,
        )
        pv_mesh_corners = non_raster_grid_to_pv_unstructured(
            grid,
            field_or_array_for_z=grid.z_of_corner,
            at="corner",
            is3d=False,
        )
    elif isinstance(grid, NetworkModelGrid):
        raise NotImplementedError(
            "Grid type " + str(type(grid)) + " is not currently handled."
        )
    else:
        pv_mesh_nodes = non_raster_grid_to_pv_unstructured(
            grid,
            field_or_array_for_z=field_for_node_z,
            base_vals=values_for_node_base,
            at="node",
            is3d=make3d,
        )
        pv_mesh_corners = non_raster_grid_to_pv_unstructured(
            grid,
            field_or_array_for_z=field_for_corner_z,
            base_vals=values_for_corner_base,
            at="corner",
            is3d=make3d,
        )

    return pv_mesh_nodes, pv_mesh_corners


def _get_pv_cell_type(number_of_vertices):
    """
    Return the PyVista CellType for the given number of vertices.
    """
    assert number_of_vertices > 2
    if number_of_vertices == 3:
        the_type = pv.CellType.TRIANGLE
    elif number_of_vertices == 4:
        the_type = pv.CellType.QUAD
    else:
        the_type = pv.CellType.POLYGON
    return the_type


def get_polygon_array_and_types(poly_verts):
    """

    Examples
    --------
    >>> from landlab import RadialModelGrid
    >>> grid = RadialModelGrid(n_rings=2, nodes_in_first_ring=5)
    >>> parray, ptypes = get_polygon_array_and_types(grid.nodes_at_patch)
    >>> parray[:8]
    array([3, 4, 0, 1, 3, 3, 4, 1])
    >>> ptypes[0]
    <CellType.TRIANGLE: 5>
    >>> parray, ptypes = get_polygon_array_and_types(grid.corners_at_cell)
    >>> parray[7:14]
    array([ 6,  5, 10,  9,  6,  1,  3])
    >>> parray[14:20]
    array([ 5, 11, 14, 10,  5,  8])
    >>> ptypes[0]
    <CellType.POLYGON: 7>
    """
    n_verts = np.count_nonzero(poly_verts + 1, axis=1)
    array_len = np.sum(n_verts) + len(poly_verts)
    poly_array = np.zeros(array_len, dtype=int)
    type_array = np.zeros(len(poly_verts), dtype=pv.CellType)
    index = 0
    for poly in range(len(poly_verts)):
        poly_array[index] = n_verts[poly]
        index += 1
        these_listed_verts = poly_verts[poly]
        these_actual_verts = these_listed_verts[these_listed_verts >= 0]
        nv = len(these_actual_verts)
        poly_array[index : index + nv] = these_actual_verts
        index += nv
        type_array[poly] = _get_pv_cell_type(nv)
    return poly_array, type_array


def non_raster_grid_to_pv_unstructured(
    grid, field_or_array_for_z, at, base_vals=None, is3d=False
):
    """
    Create and return a pyVista UntructuredGrid. Used for non-raster
    grid types such as HexModelGrid, RadialModelGrid, and IcosphereGlobalGrid.

    Parameters
    ----------
    #UPDATE
    grid : RasterModelGrid
        The grid to translate
    field_or_array_for_z : str, array, or float
        Name of field, or array, or single value to use for z coordinate
    at : str (optional)
        Which points to use: "node" (default) or "corner"
    base_vals : str, array, or float (optional; default lowest - 1/2 max wid)
        Field name, array, or single value for the z of the bottom mesh layer.

    Returns #UPDATE
    -------
    PyVista StructuredGrid object of dimensions (nr, nc, 2),
    where nr and nc are the number of node (or corner) rows
    and columns, respectively, in the Landlab grid.

    Examples #UPDATE
    --------
    >>> from landlab import RadialModelGrid
    >>> grid = RadialModelGrid(2, 5)
    >>> z = grid.add_field("z", np.arange(grid.number_of_nodes), at="node")
    >>> non_raster_grid_to_pv_unstructured(grid, "z", at="node")
    UnstructuredGrid (...)
      N Cells:    20
      N Points:   16
      X Bounds:   -2.000e+00, 2.000e+00
      Y Bounds:   -1.902e+00, 1.902e+00
      Z Bounds:   0.000e+00, 1.500e+01
      N Arrays:   1

    >>> zc = grid.add_field(
    ...     "zc", np.arange(grid.number_of_corners), at="corner")
    >>> non_raster_grid_to_pv_unstructured(grid, "zc", at="corner")
    UnstructuredGrid (...)
      N Cells:    6
      N Points:   20
      X Bounds:   -1.500e+00, 1.500e+00
      Y Bounds:   -1.577e+00, 1.577e+00
      Z Bounds:   0.000e+00, 1.900e+01
      N Arrays:   1

    """
    if at == "node":
        x = grid.x_of_node
        y = grid.y_of_node
        z = _get_z_node_vals(grid, field_or_array_for_z)
        poly_verts = grid.nodes_at_patch

    elif at == "corner":
        x = grid.x_of_corner
        y = grid.y_of_corner
        z = _get_z_corner_vals(grid, field_or_array_for_z)
        poly_verts = grid.corners_at_cell
    else:
        raise (ValueError, "'at' must be 'node' or 'corner'")

    pts = np.column_stack((x, y, z))
    poly_array, poly_types = get_polygon_array_and_types(poly_verts)

    if is3d:
        # UPDATE
        if base_vals is None:
            z_base = _set_default_base_z(x, y, z)
        elif at == "node":
            z_base = _get_z_node_vals(grid, base_vals)
        else:
            z_base = _get_z_corner_vals(grid, base_vals)
        bottom = pts.copy()
        bottom[:, 2] = z_base
        pts = np.vstack((pts, bottom))

    ug = pv.UnstructuredGrid(poly_array, poly_types, pts)

    add_fields_to_pv_dataset(grid, ug, at=at)

    return ug


@use_field_name_array_or_value("node")
def _get_reshaped_node_xyz(grid, vals):
    """
    Return a 2d array containing node (x,y,z) values for a Landlab grid.

    z values are taken from either a named field, an array, or a float.

    Parameters
    ----------
    grid : Landlab grid object
        The grid
    vals : str, array, or float
        Field name, array, or constant value for z

    Returns
    -------
    array, array, array : x, y, and z values
    """
    nr = grid.number_of_node_rows
    nc = grid.number_of_node_columns
    x = grid.x_of_node.reshape((nr, nc))
    y = grid.y_of_node.reshape((nr, nc))
    z = vals.reshape((nr, nc))
    return x, y, z


@use_field_name_array_or_value("corner")
def _get_reshaped_corner_xyz(grid, vals):
    """
    Return a 2d array containing corner (x,y,z) values for a Landlab grid.

    z values are taken from either a named field, an array, or a float.

    Parameters
    ----------
    grid : Landlab grid object
        The grid (must have defined corners)
    vals : str, array, or float
        Field name, array, or constant value for z

    Returns
    -------
    array, array, array : x, y, and z values
    """
    nr = grid.number_of_corner_rows
    nc = grid.number_of_corner_columns
    x = grid.x_of_corner.reshape((nr, nc))
    y = grid.y_of_corner.reshape((nr, nc))
    z = vals.reshape((nr, nc))
    return x, y, z


def get_reshaped_xyz(grid, field_or_array_for_z, at="node"):
    """
    Return x, y, and z coordinates of nodes or corners reshaped as
    (number of rows, number of columns). Use specified field for z.

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to be reshaped
    field_or_array_for_z : str, array, or float
        Name of the field or array containing the z coordinate values, or constant value
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
    if at == "node":
        x, y, z = _get_reshaped_node_xyz(grid, field_or_array_for_z)
    elif at == "corner":
        x, y, z = _get_reshaped_corner_xyz(grid, field_or_array_for_z)
    else:
        raise (ValueError, "'at' must be 'node' or 'corner'")

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

    Returns
    -------
    None

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from pyvista import StructuredGrid
    >>> rmg = RasterModelGrid((3, 3))
    >>> z = rmg.add_field("z", np.arange(9))
    >>> _ = rmg.add_zeros("node_field1", at="node")
    >>> _ = rmg.add_zeros("node_field2", at="node")
    >>> pvsg = StructuredGrid(
    ...     rmg.x_of_node.reshape((3, 3)),
    ...     rmg.y_of_node.reshape((3, 3)),
    ...     z.reshape((3, 3))
    ... )
    >>> _ = add_fields_to_pv_dataset(rmg, pvsg, z_field="z")
    >>> names = pvsg.array_names.copy()
    >>> names.sort()
    >>> names
    ['node_field1', 'node_field2']
    """
    is3d = False
    if at == "node":
        npts = grid.number_of_nodes
        polyname = "patch"
    else:
        npts = grid.number_of_corners
        polyname = "cell"
    if dataset.n_points == (2 * npts):
        is3d = True
    for field in grid.fields(include="at_" + at + "*"):
        fieldname = field[field.find(":") + 1 :]
        if fieldname != z_field:
            vals = grid.field_values(fieldname, at=at)
            if is3d:
                vals = np.hstack((vals, base_vals + np.zeros(npts)))
            dataset.point_data[fieldname] = vals
    for field in grid.fields(include="at_" + polyname + "*"):
        fieldname = field[field.find(":") + 1 :]
        vals = grid.field_values(fieldname, at=polyname)
        dataset.cell_data[fieldname] = vals


def raster_grid_to_pv2d_struct(grid, field_or_array_for_z, at="node"):
    """
    Translate a RasterModelGrid into a PyVista 2D StructuredGrid.
    Includes node or corner fields as data arrays.

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to translate
    field_or_array_for_z : str, array, or float
        Name of field, or array, or single value to use for z coordinate
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
    x, y, z = get_reshaped_xyz(grid, field_or_array_for_z, at)
    struc_pv_grid = pv.StructuredGrid(x, y, z)
    add_fields_to_pv_dataset(grid, struc_pv_grid, at=at)
    return struc_pv_grid


@use_field_name_array_or_value("node")
def _get_z_node_vals(grid, vals):
    """
    Given a Landlab grid and an at-node field name, node-length array, or single value,
    return a node-length array of values.

    Parameters
    ----------
    grid : Landlab grid object (any type)
        The grid to use
    vals : str, node-length array, or float
        The field name, array, or single value to return as a node-length array
    """
    return vals


@use_field_name_array_or_value("corner")
def _get_z_corner_vals(grid, vals):
    """
    Given a Landlab grid and an at-corner field name, node-length array, or single
    value, return a corner-length array of values.

    Parameters
    ----------
    grid : Landlab grid object (any type)
        The grid to use
    vals : str, corner-length array, or float
        The field name, array, or single value to return as a corner-length array
    """
    return vals


def _set_default_base_z(x, y, z):
    """
    Return a default constant value for the depth of a 3D mesh.

    Use the lowest height minus half the widest horizontal dimension.

    Parameters
    ----------
    x, y, z : ndarrays
        Arrays containing the x, y, znd z coordinates of a mesh

    Returns
    -------
    float : lowest z minus half the span of x or y (whichever span is bigger)
    """
    return np.amin(z) - max(np.amax(x) - np.amin(x), np.amax(y) - np.amin(y)) / 2


def raster_grid_to_pv3d_struct(grid, field_or_array_for_z, at="node", base_vals=None):
    """
    Create and return a pyVista 3D StructuredGrid. Same as
    raster_grid_to_pv2d_struct except that the PyVista grid
    is 3D, with two layers: one the z-coordinate of the Landlab
    grid, and the other either a flat surface at a constant z,
    a specifies array of values for the height of the bottom
    layer, or an existing field.

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to translate
    field_or_array_for_z : str, array, or float
        Name of field, or array, or single value to use for z coordinate
    at : str (optional)
        Which points to use: "node" (default) or "corner"
    base_vals : str, array, or float (optional; default lowest - 1/2 max wid)
        Field name, array, or single value for the z of the bottom mesh layer.

    Returns
    -------
    PyVista StructuredGrid object of dimensions (nr, nc, 2),
    where nr and nc are the number of node (or corner) rows
    and columns, respectively, in the Landlab grid.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((4, 5), 10.0)
    >>> z = grid.add_field(
    ...     "z", np.arange(grid.number_of_nodes), at="node", depth=20.0)
    >>> raster_grid_to_pv3d_struct(grid, "z")
    StructuredGrid (...)
      N Cells:      12
      N Points:     40
      X Bounds:     0.000e+00, 4.000e+01
      Y Bounds:     0.000e+00, 3.000e+01
      Z Bounds:     -2.000e+01, 1.900e+01
      Dimensions:   4, 5, 2
      N Arrays:     1

    >>> zc = grid.add_field(
    ...     "zc", np.arange(grid.number_of_corners), at="corner")
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

    if at == "node":
        x = grid.x_of_node
        y = grid.y_of_node
        z = _get_z_node_vals(grid, field_or_array_for_z)
        if base_vals is None:
            z_base = _set_default_base_z(x, y, z)
        else:
            z_base = _get_z_node_vals(grid, base_vals)
        nr = grid.number_of_node_rows
        nc = grid.number_of_node_columns
    elif at == "corner":
        x = grid.x_of_corner
        y = grid.y_of_corner
        z = _get_z_corner_vals(grid, field_or_array_for_z)
        if base_vals is None:
            z_base = _set_default_base_z(x, y, z)
        else:
            z_base = _get_z_corner_vals(grid, base_vals)
        nr = grid.number_of_corner_rows
        nc = grid.number_of_corner_columns
    else:
        raise (ValueError, "'at' must be 'node' or 'corner'")

    top = np.column_stack((x, y, z))
    bottom = top.copy()
    bottom[:, 2] = z_base

    vol = pv.StructuredGrid()
    vol.points = np.vstack((top, bottom))
    vol.dimensions = [nr, nc, 2]

    add_fields_to_pv_dataset(grid, vol, at=at)

    return vol
