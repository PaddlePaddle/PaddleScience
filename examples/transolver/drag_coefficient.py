"""
Reference: https://github.com/thuml/Transolver
"""
import os

import numpy as np
import vtk
from scipy.spatial import ConvexHull
from vtk.util.numpy_support import vtk_to_numpy


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


############## calculate rectangle ##############
def calculate_pos(pos):
    hull = ConvexHull(pos[:, :2])
    A = hull.volume
    return A


############## surf area ##############
def calculate_mesh_cell_area(unstructured_grid_data):
    # Read VTK file
    poly_data, _ = unstructured_grid_data_to_poly_data(unstructured_grid_data)

    # Get the points and cells
    points = poly_data.GetPoints()
    cells = poly_data.GetPolys()

    # Initialize an array to store point areas
    cell_areas = np.zeros(cells.GetNumberOfCells())

    # Iterate through cells to calculate areas
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    id = 0
    while cells.GetNextCell(cell):
        # Check if the cell is a quadrilateral
        if cell.GetNumberOfIds() == 4:
            # Get the four vertices of the quadrilateral
            p1 = np.array(points.GetPoint(cell.GetId(0)))
            p2 = np.array(points.GetPoint(cell.GetId(1)))
            p3 = np.array(points.GetPoint(cell.GetId(2)))
            p4 = np.array(points.GetPoint(cell.GetId(3)))
            # Calculate the area of the quadrilateral
            area = 0.5 * (
                np.linalg.norm(np.cross(p2 - p1, p3 - p1))
                + np.linalg.norm(np.cross(p3 - p1, p4 - p1))
            )

            # Add the area to each vertex of the quadrilateral
            cell_areas[id] += area
            id += 1

    return cell_areas


############## velocity gradient ##############
def calculate_cell_velocity_gradient(unstructured_grid_data, velocity):
    # Create a vtkDoubleArray for velocity
    velocity_data = vtk.vtkDoubleArray()
    velocity_data.SetNumberOfComponents(3)  # Assuming 3D velocity field
    velocity_data.SetNumberOfTuples(unstructured_grid_data.GetNumberOfPoints())
    velocity_data.SetName("Velocity")  # Replace "Velocity" with the desired array name

    # Set the velocity array values
    for i in range(unstructured_grid_data.GetNumberOfPoints()):
        velocity_data.SetTuple(i, velocity[i])

    # Add the velocity array to the point data
    unstructured_grid_data.GetPointData().AddArray(velocity_data)

    # Get the points and cell data (assuming velocity is stored as point data)
    poly_data, _ = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    points = poly_data.GetPoints()

    # Initialize arrays to store velocity gradients
    grad_u = np.zeros((poly_data.GetNumberOfCells(), 3))  # Assuming 3D velocity field
    # Iterate through cells to calculate gradients
    cells = poly_data.GetPolys()
    cells.InitTraversal()
    cell = vtk.vtkIdList()
    id = 0
    while cells.GetNextCell(cell):
        # Check if the cell is a quadrilateral
        if cell.GetNumberOfIds() == 4:
            # Get the four vertices of the quadrilateral
            p1 = np.array(points.GetPoint(cell.GetId(0)))
            p2 = np.array(points.GetPoint(cell.GetId(1)))
            p3 = np.array(points.GetPoint(cell.GetId(2)))
            p4 = np.array(points.GetPoint(cell.GetId(3)))
            # Calculate the velocity at each vertex
            u1 = np.array(
                poly_data.GetPointData().GetArray("Velocity").GetTuple(cell.GetId(0))
            )
            u2 = np.array(
                poly_data.GetPointData().GetArray("Velocity").GetTuple(cell.GetId(1))
            )
            u3 = np.array(
                poly_data.GetPointData().GetArray("Velocity").GetTuple(cell.GetId(2))
            )
            u4 = np.array(
                poly_data.GetPointData().GetArray("Velocity").GetTuple(cell.GetId(3))
            )

            # Calculate the gradients using finite differences
            du_dx = (u2 - u1 + u3 - u4) / (np.linalg.norm(p2 - p1 + p3 - p4) + 1e-8)
            du_dy = (u3 - u1 + u4 - u2) / (np.linalg.norm(p3 - p1 + p4 - p2) + 1e-8)
            du_dz = (u4 - u1 + u2 - u3) / (np.linalg.norm(p4 - p1 + p2 - p3) + 1e-8)

            # Add the gradients to each vertex of the quadrilateral
            grad_u[id] += du_dx + du_dy + du_dz
            id += 1

    return grad_u


############## calculate drag ##############
def calculate_drag_force(
    cell_areas, surface_normals, pressure_array, velocity_gradients, dynamic_viscosity
):
    # Calculate the pressure force component along the flow direction
    pressure_force_component = -np.dot(
        pressure_array.flatten() * cell_areas.flatten(), surface_normals.flatten()
    )

    # Calculate the wall shear stress component along the flow direction
    wall_shear_stress_component = (
        -np.dot(
            velocity_gradients.flatten() * cell_areas.flatten(),
            surface_normals.flatten(),
        )
        * dynamic_viscosity
    )
    # Sum the pressure force and wall shear stress components to get the total drag force
    drag_force = np.sum(pressure_force_component + wall_shear_stress_component)

    return drag_force


############## calculate norm ##############
def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(
        unstructured_grid_data
    )
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    return vtk_to_numpy(normal_filter.GetOutput().GetCellData().GetNormals())


############## calculate coefficient ##############
def cal_coefficient(file_name, press_surf=None, velo_surf=None):
    root = (
        "/work/EquationFlow/Transolver/Car-Design-ShapeNetCar/mlcfd_data/training_data"
    )
    # save_path = (
    #     "/work/EquationFlow/Transolver/Car-Design-ShapeNetCar/mlcfd_data/preprocessed_data/param0/"
    #     + file_name
    # )
    file_name_press = "param0/" + file_name + "/quadpress_smpl.vtk"
    file_name_velo = "param0/" + file_name + "/hexvelo_smpl.vtk"
    file_name_press = os.path.join(root, file_name_press)
    file_name_velo = os.path.join(root, file_name_velo)
    unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
    unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)

    # normal
    normal_surf = get_normal(unstructured_grid_data_press)
    # front area
    points_surf = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())
    A = calculate_pos(points_surf)
    # mesh area
    cell_areas = calculate_mesh_cell_area(unstructured_grid_data_press)
    # mesh velo
    if velo_surf is None:
        velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
        points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
        velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}
        velo_surf = np.array(
            [
                velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)
                for p in points_surf
            ]
        )
    # gradient u
    grad_u = calculate_cell_velocity_gradient(unstructured_grid_data_press, velo_surf)
    # press
    if press_surf is None:
        c2p = vtk.vtkPointDataToCellData()
        c2p.SetInputData(unstructured_grid_data_press)
        c2p.Update()
        unstructured_grid_data_press = c2p.GetOutput()
        press_surf = vtk_to_numpy(
            unstructured_grid_data_press.GetCellData().GetScalars()
        )
    else:
        # Create a vtkDoubleArray for press
        press_data = vtk.vtkDoubleArray()
        press_data.SetNumberOfComponents(1)  # Assuming 3D velocity field
        press_data.SetNumberOfTuples(unstructured_grid_data_press.GetNumberOfPoints())
        press_data.SetName("my_press")  # Replace "my_press" with the desired array name

        # Set the velocity array values
        for i in range(unstructured_grid_data_press.GetNumberOfPoints()):
            press_data.SetTuple(i, press_surf[i])

        # Add the velocity array to the point data
        unstructured_grid_data_press.GetPointData().AddArray(press_data)
        c2p = vtk.vtkPointDataToCellData()
        c2p.SetInputData(unstructured_grid_data_press)
        c2p.Update()
        unstructured_grid_data_press = c2p.GetOutput()
        press_surf = vtk_to_numpy(
            unstructured_grid_data_press.GetCellData().GetArray("my_press")
        )

    drag_force = calculate_drag_force(
        cell_areas, normal_surf[:, -1], press_surf, grad_u[:, -1], np.array(1.8e-5)
    )
    nu = 72 / 3.6
    air_density = 0.3
    cd = (2 / ((nu**2) * A * air_density)) * drag_force
    return cd
