# standard imports
import os
import numpy as np
import pandas as pd
from os.path import join

# pytorch imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# atom3d imports
from atom3d.datasets.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_center, gen_rot_matrix
from torch.utils.data import DataLoader2
import atom3d.datasets as da

# class imports
from element import Element

# element data
HYDROGEN = Element('H', 25, 2.20)
CARBON = Element('C', 70, 2.55)
NITROGEN = Element('N', 65, 3.04)
OXYGEN = Element('O', 60, 3.44)
FLUORINE = Element('F', 50, 3.98)
CHLORINE = Element('Cl', 100, 3.16)

# constants
UNRECOGNIZED = tuple([torch.zeros(1, 1, 1, 1),-1,-1]) # "Feature" element of the tensor should be a 4D tensor stuffed with 0s

# download dataset
TARGET_PATH = join(os.path.dirname(os.path.realpath(__file__)), "datasets/")
if not os.listdir(TARGET_PATH):
    da.download_dataset('lba', TARGET_PATH)

def calculate_padding(tup, max_size):
    assert len(tup) == len(max_size)
    print(tup)
    tup = tup[::-1]
    diff = tuple([max_size[i] - tup[i] for i in range(0,len(tup))])
    return tuple([diff[int(i/2)] if i%2==0 else 0 for i in range(0,len(tup)*2)])

def collate_fn(data):
    grids, labels, ids = zip(*data)
    largest_dim = max([grid.size() for grid in grids])
    print(largest_dim)
    padded_grids = tuple([F.pad(input=grid, pad=calculate_padding(grid.size(), largest_dim), mode='constant', value=0) for grid in grids])
    print(type(padded_grids[1]))
    return [(padded_grids[i], labels[i], ids[i]) for i in range(0, len(padded_grids))]

class UnrecognizedAtom(ValueError):
    pass

class DipoleDataTransform:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.grid_config = dotdict({
            'elements': {
                'H': HYDROGEN,
                'C': CARBON,
                'N': NITROGEN,
                'O': OXYGEN,
                'F': FLUORINE,
                'Cl': CHLORINE
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
    
    def _voxelize(self, atoms):
        # Take in Atom3D molecule dataframe and return custom 4D numpy array with EN and atomic radius data.
        # ------------------------------------------------------------------------------------------------
        # Prune out ANY molecule with unrecognized elements
        recognized = lambda atom: atom in self.grid_config.elements.keys()
        atoms = atoms[['x', 'y', 'z', 'element']]
        if (atoms['element'].apply(lambda item: False if recognized(item) else True).all()):
            raise UnrecognizedAtom(f"Unrecognized atom in structure--atom must be one of {self.grid_config.keys()}")

        # Cull out unwanted columns
        coords = atoms[['x', 'y', 'z']]
        atom_names = atoms['element']

        # Center dataframe around molecule
        mol_center = get_center(coords)
        centered_coords = coords - mol_center

        # Generate random rotation matrix and rotate coordinates
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        final_coords = np.dot(centered_coords, rot_mat)

        # Take in names and build an array with added EN and atomic radius data
        en = np.vectorize(lambda name: self.grid_config.elements[name].en)
        ra = np.vectorize(lambda name: self.grid_config.elements[name].ra)
        atom_data = pd.DataFrame()
        atom_data['en'] = pd.Series(en(atom_names))
        atom_data['ra'] = pd.Series(ra(atom_names))

        # Set dimensions of final grid
        xrange = round(max(final_coords.T[0]) - min(final_coords.T[0]))
        yrange = round(max(final_coords.T[1]) - min(final_coords.T[1]))
        zrange = round(max(final_coords.T[2]) - min(final_coords.T[2]))
        margin = 2
        size = (xrange + margin, yrange + margin, zrange + margin, 3)

        # Round coords and align them to zero
        final_coords = final_coords.T
        xfix = np.vectorize(lambda x: round(x + min(final_coords[0])))
        yfix = np.vectorize(lambda y: round(y + min(final_coords[1])))
        zfix = np.vectorize(lambda z: round(z + min(final_coords[2])))
        final_coords[0] = xfix(final_coords[0])
        final_coords[1] = yfix(final_coords[1])
        final_coords[2] = zfix(final_coords[2])
        final_coords = final_coords.astype(int)

        # Form final grid
        voxelized = np.zeros(size, dtype=np.float32)
        for xcoord in final_coords[0]:
            for ycoord in final_coords[1]:
                for zcoord, _en, _ra in zip(final_coords[2], atom_data['en'], atom_data['ra']):
                    voxelized[xcoord, ycoord, zcoord, 0] = 1
                    voxelized[xcoord, ycoord, zcoord, 1] = _en
                    voxelized[xcoord, ycoord, zcoord, 2] = _ra

        print("done")
        print(torch.from_numpy(voxelized).dim())
        return torch.from_numpy(voxelized)


    def __call__(self, item):
        # Transform molecule into voxel grids.
        # Apply random rotation matrix.
        try:
            transformed = tuple([ \
                # Feature
                self._voxelize(item['atoms']), \
                # Label (Dipole moment)
                item['labels'][3], \
                # ID
                int(item['id'].replace('gdb_','')) \
            ])
        except UnrecognizedAtom:
            transformed = UNRECOGNIZED
        print(type(transformed))
        return transformed
        
if __name__== "__main__":
    dataset_path = join(os.path.dirname(os.path.realpath(__file__)), "datasets/smp-random/data/test")
    dataset = LMDBDataset(dataset_path, transform=DipoleDataTransform())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    for item in dataloader:
        print(type(item))
        print(type(item[0]))
        print('feature shape:', item[0][0].shape)
        print('label:', item[0][1])
        break
