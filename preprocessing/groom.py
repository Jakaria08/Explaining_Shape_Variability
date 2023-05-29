import os
import argparse
import shapeworks as sw

from tqdm import tqdm
from glob import glob
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack


@contextmanager
def suppress_stdout(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def mkdir(path: str):
    """Checks whether directory already exists or not and creates it if not."""

    if not os.path.exists(path):
        os.makedirs(path)


def start_grooming(args):
    """Start data grooming pipeline."""
    mesh_dict = {}

    for mesh_data in tqdm((glob(f"{args.data_dir}*.ply"))):
        with suppress_stdout():
            # Extract filename
            mesh_data = os.path.normpath(mesh_data)
            filename = os.path.split(mesh_data)[-1].split(".")[0]
            #print(mesh_data)
            #print("\n")
            #print(filename)
            
            # Read binary segmentation file
            mesh_shape = sw.Mesh(mesh_data)

            # Center mesh
            center = mesh_shape.center()
            mesh_shape.translate(list(-center))

            # Write file
            path = f"{args.output_dir}after_centering/"
            mkdir(path)
            mesh_shape.write(f"{path}{filename}.ply")

            # Scale mesh
            bb = mesh_shape.boundingBox()
            bb.max.max() - bb.min.min()
            scale_factor = 1 / (bb.max.max() - bb.min.min())
            mesh_shape = mesh_shape.scale([scale_factor, scale_factor, scale_factor])

            # Write file
            path = f"{args.output_dir}after_scaling/"
            mkdir(path)
            mesh_shape.write(f"{path}{filename}.ply")

            # Append mesh to list
            mesh_dict[filename] = mesh_shape
            
    '''        
    # Find reference medoid shape
    print("[INFO] Looking for reference (medoid) shape ...")
    mesh_list = list(mesh_dict.values())
    ref_index = sw.find_reference_mesh_index(mesh_list)
    ref_mesh = mesh_list[ref_index]
    ref_name = list(mesh_dict.keys())[ref_index]
    print(f"[INFO] Reference (medoid) shape found: {ref_name}")

    # Write file
    path = f"{args.output_dir}reference_medoid_shape/"
    mkdir(path)
    ref_mesh.write(f"{path}{ref_name}.ply")
    '''
    # Align all meshes to the reference medoid shape
    ref_mesh = mesh_dict["torus_bump_template"]
    path = f"{args.output_dir}after_alignment/"
    mkdir(path)
    for name, mesh in tqdm(mesh_dict.items()):
        with suppress_stdout():
            # compute rigid transformation
            rigid_transform = mesh.createTransform(ref_mesh, sw.Mesh.AlignmentType.Rigid, 100)
            # apply rigid transform
            mesh.applyTransform(rigid_transform)
            mesh.write(f"{path}{name}.ply")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Grooming Pipeline')
    parser.add_argument('--data_dir', type=str, default='torus_bump_3/torus_bump_random_500_thickness/')
    parser.add_argument('--output_dir', type=str, default='groomed_data/')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start_grooming(args)
