import openmesh as om
import pandas as pd
from pathlib import Path
from datasets import CoMA


class MeshData(object):
    def __init__(self,
                 root,
                 template_fp,
                 split='interpolation',
                 test_exp='bareteeth',
                 transform=None,
                 pre_transform=None):
        self.root = root
        self.template_fp = template_fp
        self.split = split
        self.test_exp = test_exp
        self.transform = transform
        self.pre_transform = pre_transform
        self.train_dataset = None
        self.test_dataste = None
        self.template_points = None
        self.template_face = None
        self.mean = None
        self.std = None
        self.num_nodes = None

        self.load()

    def load(self):
        self.train_dataset = CoMA(self.root,
                                  data_split="train",
                                  split=self.split,
                                  test_exp=self.test_exp,
                                  transform=self.transform,
                                  pre_transform=self.pre_transform)

        self.val_dataset = CoMA(self.root,
                                data_split="val",
                                split=self.split,
                                test_exp=self.test_exp,
                                transform=self.transform,
                                pre_transform=self.pre_transform)

        self.test_dataset = CoMA(self.root,
                                 data_split="test",
                                 split=self.split,
                                 test_exp=self.test_exp,
                                 transform=self.transform,
                                 pre_transform=self.pre_transform)

        tmp_mesh = om.read_trimesh(self.template_fp)
        self.template_points = tmp_mesh.points()
        self.template_face = tmp_mesh.face_vertex_indices()
        self.num_nodes = self.train_dataset[0].num_nodes

        self.num_train_graph = len(self.train_dataset)
        self.num_val_graph = len(self.val_dataset)
        self.num_test_graph = len(self.test_dataset)
        self.mean = self.train_dataset.data.x.view(self.num_train_graph, -1, 3).mean(dim=0)
        self.std = self.train_dataset.data.x.view(self.num_train_graph, -1, 3).std(dim=0)
        self.save_normalization_values()
        self.normalize()

    def normalize(self):
        print('Normalizing...')
        self.train_dataset.data.x = (
            (self.train_dataset.data.x.view(self.num_train_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        self.val_dataset.data.x = (
            (self.val_dataset.data.x.view(self.num_val_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)        
        self.test_dataset.data.x = (
            (self.test_dataset.data.x.view(self.num_test_graph, -1, 3) -
             self.mean) / self.std).view(-1, 3)
        print('Done!')

    def save_normalization_values(self):
        save_dir = Path(self.root) / 'saved_normalization_values'
        save_dir.mkdir(parents=True, exist_ok=True)

        mean_np = self.mean.detach().cpu().numpy()
        std_np = self.std.detach().cpu().numpy()

        if mean_np.shape != std_np.shape:
            raise RuntimeError(f'mean/std shape mismatch: {mean_np.shape} vs {std_np.shape}')

        num_vertices = mean_np.shape[0]
        df = pd.DataFrame({
            'vertex_index': list(range(num_vertices)),
            'mean_x': mean_np[:, 0],
            'mean_y': mean_np[:, 1],
            'mean_z': mean_np[:, 2],
            'std_x': std_np[:, 0],
            'std_y': std_np[:, 1],
            'std_z': std_np[:, 2],
        })

        out_path = save_dir / 'mean_std_per_vertex.csv'
        df.to_csv(out_path, index=False)
        print(f'Saved normalization CSV: {out_path}')

    def save_mesh(self, fp, x):
        x = x * self.std + self.mean
        om.write_mesh(fp, om.TriMesh(x.numpy(), self.template_face))
