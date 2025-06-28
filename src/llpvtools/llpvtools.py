"""
llpvtools: utilities for translating Landlab grids and fields
into pyVista data structures for plotting.

The main function is grid_to_pv(). Usage and examples are shown in its
header docstring below.

Greg Tucker, University of Colorado Boulder, USA.
Original development: June 2025
"""

import numpy as np
import pyvista as pv
from landlab import RasterModelGrid, IcosphereGlobalGrid, NetworkModelGrid
from landlab.utils.decorators import use_field_name_array_or_value


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
        Values for node z coords (default "topographic__elevation" if exists, or 0)
    field_for_corner_z : str, ndarray, or float (optional)
        Values for corner z coords (default "topographic__elevation" if exists, or 0)
    make3d : bool (optional)
        Option to make a 3D mesh, with a top and bottom layer (default False)
    values_for_node_base : str, ndarray, or float (optional)
        Values for z of lower node layer (default 1/2 a grid width below lowest top z)
    values_for_corner_base : str, ndarray, or float (optional)
        Values for z of lower corner layer (default 1/2 a grid width below lowest top z)

    Returns
    -------
    PyVista.StructuredGrid, (PyVista.StructuredGrid or None)
        A StructuredGrid for nodes, one for corners (or None if no defined corners)

    Examples
    --------

    Raster grid (note alternative possible usage for node and corner z-coordinates):

    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((4, 5), 10.0)
    >>> z = grid.add_field(
    ...     "z", np.arange(grid.number_of_nodes), at="node")
    >>> zc = grid.add_field(
    ...     "zc", np.arange(grid.number_of_corners), at="corner")
    >>> pv_sg_node, pv_sg_cnr = grid_to_pv(
    ...     grid, field_for_node_z="z", field_for_corner_z=zc
    ... )
    >>> pv_sg_node
    StructuredGrid (...)
      N Cells:      12
      N Points:     20
      X Bounds:     0.000e+00, 4.000e+01
      Y Bounds:     0.000e+00, 3.000e+01
      Z Bounds:     0.000e+00, 1.900e+01
      Dimensions:   4, 5, 1
      N Arrays:     1

    >>> pv_sg_cnr
    StructuredGrid (...)
      N Cells:      6
      N Points:     12
      X Bounds:     5.000e+00, 3.500e+01
      Y Bounds:     5.000e+00, 2.500e+01
      Z Bounds:     0.000e+00, 1.100e+01
      Dimensions:   3, 4, 1
      N Arrays:     1

    Raster grid translated to 3D:

    >>> pv_sg_node, pv_sg_cnr = grid_to_pv(
    ...     grid, field_for_node_z="z", make3d=True, values_for_node_base=-5.0
    ... )
    >>> pv_sg_node
    StructuredGrid (...)
      N Cells:      12
      N Points:     40
      X Bounds:     0.000e+00, 4.000e+01
      Y Bounds:     0.000e+00, 3.000e+01
      Z Bounds:     -5.000e+00, 1.900e+01
      Dimensions:   4, 5, 2
      N Arrays:     1

    Unstructured grid examples:

    >>> from landlab import RadialModelGrid
    >>> grid = RadialModelGrid(n_rings=2, nodes_in_first_ring=5)
    >>> z = grid.add_field("z", np.arange(grid.number_of_nodes), at="node")
    >>> zc = grid.add_field(
    ...     "zc", np.arange(grid.number_of_corners), at="corner")
    >>> pv_ug_n, pv_ug_c = grid_to_pv(grid, field_for_node_z=z, field_for_corner_z=zc)
    >>> pv_ug_n
    UnstructuredGrid (...)
      N Cells:    20
      N Points:   16
      X Bounds:   -2.000e+00, 2.000e+00
      Y Bounds:   -1.902e+00, 1.902e+00
      Z Bounds:   0.000e+00, 1.500e+01
      N Arrays:   1
    >>> pv_ug_c
    UnstructuredGrid (...)
      N Cells:    6
      N Points:   20
      X Bounds:   -1.500e+00, 1.500e+00
      Y Bounds:   -1.577e+00, 1.577e+00
      Z Bounds:   0.000e+00, 1.900e+01
      N Arrays:   1

    Unstructured as 3D:

    >>> pv_ug_n, pv_ug_c = grid_to_pv(
    ...     grid, field_for_node_z=z, field_for_corner_z=zc, make3d=True
    ... )
    >>> pv_ug_n
    UnstructuredGrid (...)
      N Cells:    20
      N Points:   32
      X Bounds:   -2.000e+00, 2.000e+00
      Y Bounds:   -1.902e+00, 1.902e+00
      Z Bounds:   -2.000e+00, 1.500e+01
      N Arrays:   1
    >>> pv_ug_c
    UnstructuredGrid (...)
      N Cells:    6
      N Points:   40
      X Bounds:   -1.500e+00, 1.500e+00
      Y Bounds:   -1.577e+00, 1.577e+00
      Z Bounds:   -1.577e+00, 1.900e+01
      N Arrays:   1

    NetworkModelGrid example:

    >>> from landlab import NetworkModelGrid
    >>> y_of_node = (0, 1, 2, 2)
    >>> x_of_node = (0, 0, -1, 1)
    >>> nodes_at_link = ((1, 0), (2, 1), (3, 1))
    >>> nmg = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
    >>> z = nmg.add_field("example_node_data", 0.1 * np.arange(4), at="node")
    >>> _ = nmg.add_field("example_link_data", np.arange(3), at="link")
    >>> pvug = network_grid_to_pv_unstructured(nmg, z)
    >>> pvug
    UnstructuredGrid (...)
      N Cells:    3
      N Points:   4
      X Bounds:   -1.000e+00, 1.000e+00
      Y Bounds:   0.000e+00, 2.000e+00
      Z Bounds:   0.000e+00, 3.000e-01
      N Arrays:   2
    >>> pvug.point_data["example_node_data"]
    pyvista_ndarray([0. , 0.1, 0.2, 0.3])
    >>> pvug.cell_data["example_link_data"]
    pyvista_ndarray([0, 1, 2])

    IcoSphereGlobalGrid example:

    >>> from landlab import IcosphereGlobalGrid
    >>> ico = IcosphereGlobalGrid()
    >>> pv_ug_n, pv_ug_c = grid_to_pv(ico)
    >>> pv_ug_n
    UnstructuredGrid (...)
      N Cells:    20
      N Points:   12
      X Bounds:   -8.507e-01, 8.507e-01
      Y Bounds:   -8.507e-01, 8.507e-01
      Z Bounds:   -8.507e-01, 8.507e-01
      N Arrays:   0
    >>> pv_ug_c
    UnstructuredGrid (...)
      N Cells:    12
      N Points:   20
      X Bounds:   -9.342e-01, 9.342e-01
      Y Bounds:   -9.342e-01, 9.342e-01
      Z Bounds:   -9.342e-01, 9.342e-01
      N Arrays:   0
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
        pv_mesh_nodes = network_grid_to_pv_unstructured(grid, field_for_node_z)
        pv_mesh_corners = None
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


def raster_grid_to_pv3d_struct(grid, field_or_array_for_z, at="node", base_vals=None):
    """
    Create and return a pyVista 3D StructuredGrid. Same as
    raster_grid_to_pv2d_struct except that the PyVista grid
    is 3D, with two layers: one the z-coordinate of the Landlab
    grid, and the other either a flat surface at a constant z,
    a specified array of values for the height of the bottom
    layer, or an existing field.

    Parameters
    ----------
    grid : RasterModelGrid
        The grid to translate
    field_or_array_for_z : str, array, or float
        Name of field, or array, or single value to use for z coordinate
    at : str (optional)
        Which points to use: "node" (default) or "corner"
    base_vals : str, array, or float (optional; default lowest - 1/2 max width)
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
    >>> raster_grid_to_pv3d_struct(grid, "z", at="node")
    StructuredGrid (...)
      N Cells:      12
      N Points:     40
      X Bounds:     0.000e+00, 4.000e+01
      Y Bounds:     0.000e+00, 3.000e+01
      Z Bounds:     -2.000e+01, 1.900e+01
      Dimensions:   4, 5, 2
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

    _add_fields_to_pv_dataset(grid, vol, at=at)

    return vol


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
    x, y, z = _get_reshaped_xyz(grid, field_or_array_for_z, at)
    struc_pv_grid = pv.StructuredGrid(x, y, z)
    _add_fields_to_pv_dataset(grid, struc_pv_grid, at=at)
    return struc_pv_grid


def non_raster_grid_to_pv_unstructured(
    grid, field_or_array_for_z, at, base_vals=None, is3d=False
):
    """
    Create and return a pyVista UnstructuredGrid. Used for non-raster
    grid types such as HexModelGrid, RadialModelGrid, and IcosphereGlobalGrid.

    Parameters
    ----------
    grid : Landlab non-raster grid
        The grid to translate
    field_or_array_for_z : str, array, or float
        Name of field, or array, or single value to use for z coordinate
    at : str
        Which Landlab mesh type to translate: either "node" or "corner"
    base_vals : str, array, or float (optional; default lowest - 1/2 max width)
        Field name, array, or single value for z of bottom mesh layer (if is3d==True)
    is3d : bool (optional)
        If True, makes a 3D mesh with base_vals used for z at the bottom (default False)

    Returns
    -------
    PyVista UnstructuredGrid object

    Examples
    --------
    >>> from landlab import RadialModelGrid
    >>> grid = RadialModelGrid(n_rings=2, nodes_in_first_ring=5)
    >>> z = grid.add_field("z", np.arange(grid.number_of_nodes), at="node")
    >>> non_raster_grid_to_pv_unstructured(grid, "z", at="node", is3d=True)
    UnstructuredGrid (...)
      N Cells:    20
      N Points:   32
      X Bounds:   -2.000e+00, 2.000e+00
      Y Bounds:   -1.902e+00, 1.902e+00
      Z Bounds:   -2.000e+00, 1.500e+01
      N Arrays:   1
    """
    if at == "node":
        x = grid.x_of_node
        y = grid.y_of_node
        z = _get_z_node_vals(grid, field_or_array_for_z)
        pvcell_verts = grid.nodes_at_patch
        if is3d:
            if base_vals is None:
                base_vals = _set_default_base_z(x, y, z)
            x = np.hstack((x, x))
            y = np.hstack((y, y))
            z = np.hstack((z, _get_z_node_vals(grid, base_vals)))
            pvcell_verts = _add_vertices_to_base_for_3d(
                pvcell_verts, grid.number_of_nodes
            )
    elif at == "corner":
        x = grid.x_of_corner
        y = grid.y_of_corner
        z = _get_z_corner_vals(grid, field_or_array_for_z)
        pvcell_verts = grid.corners_at_cell
        if is3d:
            if base_vals is None:
                base_vals = _set_default_base_z(x, y, z)
            x = np.hstack((x, x))
            y = np.hstack((y, y))
            z = np.hstack((z, _get_z_corner_vals(grid, base_vals)))
            pvcell_verts = _add_vertices_to_base_for_3d(
                pvcell_verts, grid.number_of_corners
            )
    else:
        raise (ValueError, "'at' must be 'node' or 'corner'")

    pts = np.column_stack((x, y, z))
    pvcell_array, pvcell_types = _get_pvcell_array_and_types(pvcell_verts, is3d)

    ug = pv.UnstructuredGrid(pvcell_array, pvcell_types, pts)

    _add_fields_to_pv_dataset(grid, ug, at=at)

    return ug


def network_grid_to_pv_unstructured(grid, field_or_array_for_z):
    """
    Translate a Landlab NetworkModelGrid to a PyVista unstructured grid made of
    line segments.

    Examples
    --------
    >>> from landlab import NetworkModelGrid
    >>> y_of_node = (0, 1, 2, 2)
    >>> x_of_node = (0, 0, -1, 1)
    >>> nodes_at_link = ((1, 0), (2, 1), (3, 1))
    >>> nmg = NetworkModelGrid((y_of_node, x_of_node), nodes_at_link)
    >>> z = nmg.add_field("example_node_data", 0.1 * np.arange(4), at="node")
    >>> pvug = network_grid_to_pv_unstructured(nmg, z)
    >>> pvug
    UnstructuredGrid (...)
      N Cells:    3
      N Points:   4
      X Bounds:   -1.000e+00, 1.000e+00
      Y Bounds:   0.000e+00, 2.000e+00
      Z Bounds:   0.000e+00, 3.000e-01
      N Arrays:   1
    """
    x = grid.x_of_node
    y = grid.y_of_node
    z = _get_z_node_vals(grid, field_or_array_for_z)
    points = np.column_stack((x, y, z))
    pvcell_array = np.column_stack(
        (2 + np.zeros(grid.number_of_links, dtype=int), grid.nodes_at_link)
    ).flatten()
    pvcell_types = grid.number_of_links * [pv.CellType.LINE]
    ug = pv.UnstructuredGrid(pvcell_array, pvcell_types, points)
    for field in grid.at_node.keys():
        ug.point_data[field] = grid.at_node[field]
    for field in grid.at_link.keys():
        ug.cell_data[field] = grid.at_link[field]
    return ug


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

    Returns
    -------
    ndarray : 1d array of values
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

    Returns
    -------
    ndarray : 1d array of values
    """
    return vals


def _set_default_base_z(x, y, z):
    """
    Return a default constant value for the depth of a 3D mesh.

    Use the lowest height minus half the widest horizontal dimension.

    Parameters
    ----------
    x, y, z : ndarrays
        Arrays containing the x, y, and z coordinates of a mesh

    Returns
    -------
    float : lowest z minus half the span of x or y (whichever span is bigger)
    """
    return np.amin(z) - max(np.amax(x) - np.amin(x), np.amax(y) - np.amin(y)) / 2


def _add_fields_to_pv_dataset(grid, dataset, z_field="", at="node", base_vals=0.0):
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
    >>> _ = _add_fields_to_pv_dataset(rmg, pvsg, z_field="z")
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


def _get_reshaped_xyz(grid, field_or_array_for_z, at="node"):
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
    >>> xx, yy, zz = _get_reshaped_xyz(grid, "z")
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


def _add_vertices_to_base_for_3d(top_verts, n_pts):
    """
    Given a 2D array of vertex IDs, this function effectively turns it into
    a prismatic column by adding a set of IDs that are equal to the original
    plus the total number of points.

    For example, if you're starting with IDs for triangular patches in a
    Landlab grid, the grid attribute nodes_at_patch will be a (# of nodes, 3)
    array containing the IDs of the nodes in each triagular patch. This function
    creates a version of the original array in which there are now 6 instead of
    3 columns, representing the vertices of triangles at the "base" of a
    triangular prism. Suppose there are 16 nodes in the grid and the IDs of
    nodes in the first triangular patch are [4, 0, 1]. This function will return
    a 16 x 6 array, in which the row for the first patch is [4, 0, 1, 20, 16, 17].
    We have in effect "made up" a duplicate set of nodes that represent a new
    "base" layer of the grid, with the same (x, y) coordinates but lower z
    coordinates (though this function doesn't concern itself with x, y, z).

    Wherever a -1 appears in the original vertex array, this indicates that no
    vertex is present at the given array position. For example if you have a
    grid with 5 hexagonal cells and 1 pentagonal cell, then the corners_at_cell
    array will be 6 x 6, but the row representing the pentagonal cell will have
    a -1 as one of its entries, so that it contains only 5 valid corner IDs.
    When we add a "base layer" of corners, we need to also insert a -1 in the
    corresponding position in the base layer. So if the original corners_at_node
    array had a row with the IDs [11, 14, 10,  5,  8, -1], the new array will
    read: [11, 14, 10,  5,  8, -1, 31, 34, 30, 25, 28, -1]. This row represents
    a pentagonal prism, with the original cell representing the top, and a new
    pentagon with the same (x, y) coordinates representing the bottom, for a
    total of 10 valid vertex IDs (instead of 12).

    Parameters
    ----------
    top_verts : 2d array of int
        2d array of vertex IDs (from Landlab grid corners_at_cell or nodes_at_patch)
    n_pts : int
        Number of vertices in the grid (nodes or corners)

    Returns
    -------
    2d array : rows hold the IDs of vertices, including "top" and "base"

    Examples
    --------
    >>> from landlab import RadialModelGrid
    >>> grid = RadialModelGrid(n_rings=2, nodes_in_first_ring=5)
    >>> _add_vertices_to_base_for_3d(grid.corners_at_cell, grid.number_of_corners)
    array([[ 4,  8,  5,  3,  0,  2, 24, 28, 25, 23, 20, 22],
           [ 5, 10,  9,  6,  1,  3, 25, 30, 29, 26, 21, 23],
           [11, 14, 10,  5,  8, -1, 31, 34, 30, 25, 28, -1],
           [13, 15, 11,  8,  4,  7, 33, 35, 31, 28, 24, 27],
           [16, 17, 12,  9, 10, 14, 36, 37, 32, 29, 30, 34],
           [18, 19, 16, 14, 11, 15, 38, 39, 36, 34, 31, 35]])
    """
    base_verts = top_verts.copy() + n_pts
    base_verts[top_verts == -1] = -1
    return np.column_stack((top_verts, base_verts))


def _get_pvcell_array_and_types(pvcell_verts, is3d):
    """
    Create and return two data structures that PyVista needs to make an
    UnstructuredGrid:a 1d array of cell vertices, with each entry starting with
    the number of vertices in the cell; and an array containing the VTK/PyVista
    CellType code for each cell.

    Parameters
    ----------
    pvcell_verts : 2d array of int
        either the nodes_at_patch or corners_at_cell attribute of a Landlab grid
    is3d : bool
        if True, the function will convert the polygons into prisms

    Returns
    -------
    1d array of int
        VTK/PyVista "cell array" with # of vertices followed by vertex IDs
    1d array of CellType
        CellType for each cell

    Examples
    --------
    >>> from landlab import RadialModelGrid
    >>> grid = RadialModelGrid(n_rings=2, nodes_in_first_ring=5)
    >>> parray, ptypes = _get_pvcell_array_and_types(grid.nodes_at_patch, is3d=False)
    >>> parray[:8]
    array([3, 4, 0, 1, 3, 3, 4, 1])
    >>> ptypes[0]
    <CellType.TRIANGLE: 5>
    >>> parray, ptypes = _get_pvcell_array_and_types(grid.corners_at_cell, is3d=False)
    >>> parray[7:14]
    array([ 6,  5, 10,  9,  6,  1,  3])
    >>> parray[14:20]
    array([ 5, 11, 14, 10,  5,  8])
    >>> ptypes[0]
    <CellType.POLYGON: 7>

    The following example is 3D: cells are turned into hexagonal prisms:

    >>> verts = _add_vertices_to_base_for_3d(
    ...     grid.corners_at_cell, grid.number_of_corners
    ... )
    >>> parray, ptypes = _get_pvcell_array_and_types(verts, is3d=True)
    >>> parray[:13]
    array([12,  4,  8,  5,  3,  0,  2, 24, 28, 25, 23, 20, 22])
    """
    n_verts_per_object = np.count_nonzero(pvcell_verts + 1, axis=1)
    n_total_verts = np.sum(n_verts_per_object)
    array_len = n_total_verts + len(pvcell_verts)
    pvcell_array = np.zeros(array_len, dtype=int)
    type_array = np.zeros(len(pvcell_verts), dtype=pv.CellType)
    index = 0
    for poly in range(len(pvcell_verts)):
        pvcell_array[index] = n_verts_per_object[poly]
        index += 1
        these_listed_verts = pvcell_verts[poly]
        these_actual_verts = these_listed_verts[these_listed_verts >= 0]
        nv = len(these_actual_verts)
        pvcell_array[index : index + nv] = these_actual_verts
        index += nv
        type_array[poly] = _get_pv_cell_type(nv, is3d)
    return pvcell_array, type_array


def _get_pv_cell_type(number_of_vertices, is3d):
    """
    Return the PyVista CellType for the given number of vertices.

    The logic here is as follows:
        - The object is either a polygon (is3d==False) or a polyhedron (is3d==True)
        - Anything 2D with >4 vertices is a PyVista POLYGON
        - If it's 3D, then we assume it's a "columnar" polyhedron, i.e., same number
          of vertices on "top" and "bottom", and quadrilaterals on the "sides"
          (like a prism standing on one end)
        - If we're mapping to 3D, then the relationship between the Landlab polygon
        shape and the 3D PyVista equivalent is:
          - triangle = WEDGE (polyhedron: 2 triangular and 3 quadrilateral faces)
          - quad = HEXAHEDRON (polyhedron: 6 quadrilateral faces)
          - 5-sided = PENTAGONAL_PRISM (polyhedron: 2 pentagonal, 5 quadrilateral faces)
          - 6-sided = HEXAGONAL_PRISM (polyhedron: 2 hexagonal, 5 quadrilateral faces)
          - 7 or more sides: no 3D mapping available; throws an exception

    Parameters
    ----------
    number_of_vertices : int
        Number of vertices in the cell
    is3d : book
        Indicates whether the cell is 2D (polygon) or 3D (polyhedron)

    Return
    ------
    CellType
        The cell type code corresponding to the # of vertices and 2D/3D
    """
    assert number_of_vertices > 2
    if number_of_vertices == 3:
        the_type = pv.CellType.TRIANGLE
    elif number_of_vertices == 4:
        the_type = pv.CellType.QUAD
    elif number_of_vertices > 4 and not is3d:
        the_type = pv.CellType.POLYGON
    elif number_of_vertices == 6:
        the_type = pv.CellType.WEDGE
    elif number_of_vertices == 8:
        the_type = pv.CellType.HEXAHEDRON
    elif number_of_vertices == 10:
        the_type = pv.CellType.PENTAGONAL_PRISM
    elif number_of_vertices == 12:
        the_type = pv.CellType.HEXAGONAL_PRISM
    else:
        raise TypeError(
            str(number_of_vertices)
            + " "
            + "Cannot make a 3D PyVista object when cells or patches have\n"
            + "more than 6 points. Try using the 2D option instead."
        )
    return the_type
