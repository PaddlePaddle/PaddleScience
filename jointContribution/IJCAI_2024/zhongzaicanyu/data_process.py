import os

import numpy as np
import utils.paddle_aux  # NOQA
import vtk


def load_centroid(file_path):
    centroid = np.load(file_path).reshape((-1, 3)).astype(np.float32)
    return centroid


def load_pressure(file_path):
    press = np.load(file_path).reshape((-1,)).astype(np.float32)
    return press


def write_vtk(vertices, pressure_data, output_path):
    points = vtk.vtkPoints()
    for idx, vertex in enumerate(vertices):
        points.InsertNextPoint(vertex)
    cells = vtk.vtkCellArray()
    for idx in range(len(vertices)):
        cells.InsertNextCell(1)
        cells.InsertCellPoint(idx)
    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(points)
    unstructured_grid.SetCells(vtk.VTK_VERTEX, cells)
    pressure = vtk.vtkFloatArray()
    pressure.SetName("Pressure")
    for idx, value in enumerate(pressure_data):
        pressure.InsertNextValue(value)
    unstructured_grid.GetPointData().AddArray(pressure)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(unstructured_grid)
    writer.Write()


def process_directory(input_centroid_dir, input_pressure_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name in os.listdir(input_centroid_dir):
        if file_name.endswith(".npy") and file_name.startswith("centroid_"):
            mesh_index = file_name.replace("centroid_", "").replace(".npy", "")
            centroid_file_path = os.path.join(input_centroid_dir, file_name)
            pressure_file_name = f"press_{mesh_index}.npy"
            pressure_file_path = os.path.join(input_pressure_dir, pressure_file_name)
            vtk_file_path = os.path.join(output_dir, f"mesh_{mesh_index}.vtk")
            if os.path.exists(pressure_file_path):
                vertices = load_centroid(centroid_file_path)
                pressure_data = load_pressure(pressure_file_path)
                num_vertices = tuple(vertices.shape)[0]
                num_pressure = tuple(pressure_data.shape)[0]
                if num_pressure > num_vertices:
                    print(
                        f"Warning: Pressure data for {file_name} is larger than the number of points. ",
                        "Trimming extra data.",
                    )
                    pressure_data = pressure_data[:num_vertices]
                elif num_pressure < num_vertices:
                    print(
                        f"Warning: Pressure data for {file_name} is smaller than the number of points. ",
                        "Trimming extra points.",
                    )
                    vertices = vertices[:num_pressure]
                write_vtk(vertices, pressure_data, vtk_file_path)
                print(f"Processed {file_name} to {vtk_file_path}")
            else:
                print(f"Pressure file for {file_name} not found.")


def write_vtk2(vertices, output_path):
    points = vtk.vtkPoints()
    for idx, vertex in enumerate(vertices):
        points.InsertNextPoint(vertex)
    cells = vtk.vtkCellArray()
    for idx in range(len(vertices)):
        cells.InsertNextCell(1)
        cells.InsertCellPoint(idx)
    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(points)
    unstructured_grid.SetCells(vtk.VTK_VERTEX, cells)
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(unstructured_grid)
    writer.Write()


def process_directory2(input_centroid_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name in os.listdir(input_centroid_dir):
        if file_name.endswith(".npy") and file_name.startswith("centroid_"):
            mesh_index = file_name.replace("centroid_", "").replace(".npy", "")
            centroid_file_path = os.path.join(input_centroid_dir, file_name)
            vtk_file_path = os.path.join(output_dir, f"mesh_{mesh_index}.vtk")
            vertices = load_centroid(centroid_file_path)
            write_vtk2(vertices, vtk_file_path)
            print(f"Processed {file_name} to {vtk_file_path}")


def data_process():
    input_centroid_directory = "./Dataset/data_track_B"
    input_pressure_directory = "./Dataset/data_track_B"
    output_directory = "./Dataset/data_centroid_track_B_vtk"
    process_directory(
        input_centroid_directory, input_pressure_directory, output_directory
    )

    input_centroid_directory = "./Dataset/track_B"
    output_directory = "./Dataset/track_B_vtk"
    process_directory2(input_centroid_directory, output_directory)
