import os
import pyvista as pv

from glob import glob
from tqdm import tqdm

if not os.path.exists("groomed_data\\after_alignment\\vtk_files"):
        os.makedirs("groomed_data\\after_alignment\\vtk_files")

for mesh_path in tqdm(glob("groomed_data\\after_alignment\\*.ply")):
        filename = mesh_path.split("\\")[-1].split(".")[0]
        mesh = pv.read(mesh_path)
        mesh.save(f"groomed_data\\after_alignment\\vtk_files\\{filename}.vtk")