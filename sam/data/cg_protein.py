import os
import json
import pickle
import shutil
import re
from collections import namedtuple
import glob
from typing import Callable, Union

import numpy as np
import mdtraj

import torch

from sam.coords import sample_data
from sam.data.sequences import aa_list, aa_one_letter
# from sam.data.common import (# get_protein_data_files,
#                                  load_datasets_common,
#                                  load_enc_datasets_common,
#                                  get_dataset_arg)
# from sam.structure import slice_traj_to_com
# from sam.data.ca_protein import (get_accept_prob_rule,
#                                  Uniform_length_batch_sampler)


################################################################################
# Code for storing molecular data in the batches built in dataloaders.         #
################################################################################

# Classes derived from these namedtuples will be used to instantiate the objects
# returned by the .get() methods of the dataset classes in this module.
_StaticData = namedtuple("StaticData",
                         ["x", "a", "ae", "r", "x_t"],
                         defaults=[0, 0, 0, 0, 0])
staticdata_enc_keys = ["z", "a", "ae", "r", "z_t"]
_StaticDataEnc = namedtuple("StaticDataEnc",
                            staticdata_enc_keys,
                            defaults=[0, 0, 0, 0, 0])

class StaticDataMixin:
    """
    Class to emulate torch_geometric batches using data from regular pytorch
    batches.
    """

    def to(self, device):
        return self._replace(
            **{k: getattr(self, k).to(device) for k in self._fields})
    
    @property
    def num_graphs(self):
        return self.sel_val.shape[0]

    @property
    def num_nodes(self):
        return self.sel_val.shape[0]*self.sel_val.shape[1]

    @property
    def batch(self):
        return torch.arange(0, self.sel_val.shape[0],
                            device=self.sel_val.device)

    def select_ids(self, ids):
        return self.__class__(
            **{k: getattr(self, k)[ids] for k in self._fields})

    def __str__(self):
        cls_name = self.__class__.__name__
        obj_repr = " ".join(["{}={}".format(k, tuple(getattr(self, k).shape)) \
                             for k in self._fields])
        return "{}: {}".format(cls_name, obj_repr)

    
class StaticData(_StaticData, StaticDataMixin):
    __slots__ = ()

    @property
    def sel_val(self):
        return self.x

class StaticDataEnc(_StaticDataEnc, StaticDataMixin):
    __slots__ = ()

    @property
    def sel_val(self):
        return self.z


#-------------------------------------------------------------------------------
# Common code for molecular and encoded datasets.                              -
#-------------------------------------------------------------------------------

class CG_Protein:

    def __init__(self, name, seq, xyz,
                 r=None,
                 # aa_embeddings_dp=None
                 ):
        self.name = name
        self.seq = seq
        self.a = get_features_from_seq(seq).argmax(0)
        use_aa_embeddings = False  # aa_embeddings_dp is not None
        if use_aa_embeddings:
            # aa_embedding_fp = os.path.join(aa_embeddings_dp, self.name + ".pt")
            # if not os.path.isfile(aa_embedding_fp):
            #     raise FileNotFoundError(aa_embedding_fp)
            # self.ae = torch.load(aa_embedding_fp)[0]
            raise NotImplementedError()
        else:
            self.ae = None
        self.xyz = xyz
        self.r = r  # Residue positional ids.
        self.e = None  # Energies.
        self.enc = None  # Encodings.

        # Graph structure.
        self.edge_index = None
        self.nr_edge_index = None
        self.chain_torsion_mask = None
        

    def set_encoding(self, enc):
        self.enc = enc


def get_features_from_seq(seq):
    n_res = len(seq)
    # Feature tensor: 20 channels for aa.
    n_features = 20
    features = np.zeros((n_features, n_res))
    # Residues one hot encoding.
    for i, aa_i in enumerate(seq):
        features[aa_list.index(aa_i),i] = 1
    return features

# ################################################################################
# # Get protein trajectory file paths.                                           #
# ################################################################################

# def _check_ca_atom(a):
#     return a.name == "CA" and a.residue.code in aa_one_letter

# def _check_cg_atom(a):
#     return a.name in ("CG", "CG2") and a.residue.code in aa_one_letter


# class Protein_data_files:
#     """
#     Class for storing the paths of data files of a protein.
#     """
#     def __init__(self, input_data, re_filter_fn=None, data_type="xyz",
#                  use_enc_scaler=None):
#         # Check the input.
#         if isinstance(input_data, str):
#             if input_data.endswith(".json"):
#                 with open(input_data, "r") as i_fh:
#                     json_data = json.load(i_fh)
#             # Legacy mode.
#             elif input_data.endswith(".json.enc"):
#                 with open(input_data, "r") as i_fh:
#                     json_data = json.load(i_fh)
#                 if use_enc_scaler is None:
#                     raise ValueError()
#                 if not use_enc_scaler:
#                     fn_suffix = "enc"
#                 else:
#                     fn_suffix = "enc_std"
#                 trajectories = []
#                 for t in json_data["trajectories"]:
#                     trajectories.append("{}.{}.pt".format(t, fn_suffix))
#                 json_data["trajectories"] = trajectories
#                 # if use_enc_scaler is None:
#                 #     raise ValueError()
#                 # fn_suffix = "enc" if not use_enc_scaler else "enc_std"
#                 # with open(input_data, "rb") as i_fh:
#                 #     data = pickle.load(i_fh)
#                 #     json_data = {}
#                 #     json_data["name"] = data["name"]
#                 #     json_data["seq"] = data["seq"]
#                 #     json_data["topology"] = None
#                 #     json_data["trajectories"] = []
#                 #     for p in data["trajectories"].keys():
#                 #         json_data["trajectories"].append(
#                 #             "{}.{}.pt".format(p, fn_suffix))
#             else:
#                 raise ValueError(input_data)
#         elif isinstance(input_data, dict):
#             json_data = input_data
#         else:
#             raise TypeError("Incorrect type for 'input_data'")
#         # Name and topology file.
#         self.name = json_data["name"]
#         self.top_fp = json_data["topology"]
#         self.seq = json_data.get("seq")

#         # Get the trajectories files.
#         if not isinstance(json_data["trajectories"], (list, tuple)):
#             raise TypeError(
#                 "Incorrect type for the 'trajectories' attribute")
#         # Get the files from a list that can look like:
#         #    ["/home/user/traj_data_0/traj_0.dcd",
#         #     "/home/user/traj_data_0/traj_1.dcd",
#         #     "/home/user/traj_data_1/traj_*.dcd""]
#         self.traj_fp_l = []
#         for traj_pattern in json_data["trajectories"]:
#             for t in glob.glob(traj_pattern):
#                 self.traj_fp_l.append(t)

#         # Filter trajectories by some pattern in their file paths.
#         if re_filter_fn is not None:
#             self.traj_fp_l = [t for t in self.traj_fp_l \
#                               if re.search(re_filter_fn, t) is not None]
#         if not self.traj_fp_l:
#             raise ValueError(
#                 "No trajectories files in input data for {}".format(self.name))


# def get_protein_data_files(input_fp, proteins=None, re_filter_fn=None,
#                            data_type="xyz", use_enc_scaler=None):
#     # Get the list of selected proteins.
#     if proteins is None:
#         _proteins = None
#     elif isinstance(proteins, (list, tuple)):
#         _proteins =  proteins
#     elif isinstance(proteins, str):
#         _proteins = []
#         with open(proteins, "r") as i_fh:
#             for l in i_fh:
#                 if l.startswith("#") or not l.rstrip():
#                     continue
#                 _proteins.append(l.rstrip())
#     else:
#         raise TypeError(proteins.__class__)
#     # Each element in this list will contain an object representing a protein
#     # and all its data files.
#     prot_data_files = []
#     # Each file in the pattern must correspond to a protein.
#     for json_fp_i in glob.glob(input_fp):
#         # Gets the data files from the JSON files.
#         prot_data_files_i = Protein_data_files(input_data=json_fp_i,
#                                                re_filter_fn=re_filter_fn,
#                                                data_type=data_type,
#                                                use_enc_scaler=use_enc_scaler)
#         # Filter by protein name.
#         if _proteins is not None:
#             if prot_data_files_i.name not in _proteins:
#                 continue
#         prot_data_files.append(prot_data_files_i)
#     if not prot_data_files:
#         raise ValueError("No protein data files were found")
#     return prot_data_files

# def apply_frames_prob_rule(enc, accept_prob_rule):
#     if accept_prob_rule is not None:
#         raise NotImplementedError()
#         # accept_p = accept_prob_rule(enc.shape[1])
#         # accept = np.random.rand(enc.shape[0]) < accept_p
#         # return enc[accept]
#     else:
#         return enc

################################################################################
# Common code for xyz and encoding datasets.                                   #
################################################################################

class CG_ProteinDatasetMixin:
    """Common methods for both the xyz and encoding datasets."""

    def _init_common(self,
                     input_fp_list: list,
                     stride: int = 1,
                     n_trajs: int = None,
                     n_frames: int = None,
                     frames_mode: str = "ensemble",
                     accept_prob_rule: Callable = None,
                     proteins: Union[list, str] = None,
                     re_filter_fn: str = None,
                     input_format: str = "torch",
                     # aa_embeddings_dp: str = None,
                     tbm_mode: str = None,
                     tbm_lag: int = 1,
                     verbose: bool = False):

        if not isinstance(input_fp_list, (list, tuple)):
            raise TypeError()
                
        self.input_fp_list = input_fp_list
        self.re_filter_fn = re_filter_fn
        self._init_protein_data_files(proteins)
        
        self.input_format = input_format
        
        self.n_trajs = n_trajs
        self.n_frames = n_frames
        if not frames_mode in ("trajectory", "ensemble"):
            raise KeyError(frames_mode)
        self.frames_mode = frames_mode
        
        if stride != 1:
            raise NotImplementedError()
        self.stride = stride
        self.accept_prob_rule = accept_prob_rule

        if tbm_mode not in (None, "random", "lag"):  # "eval"
            raise KeyError(tbm_mode)
        self.tbm_mode = tbm_mode
        self.aa_embeddings_dp = None  # aa_embeddings_dp
        self.verbose = verbose

        self.selected_mols = None
        self.protein_list = None
        self.frames = None
        self._frames = None

        
    def _print(self, msg):
        if self.verbose:
            print(msg)

    def _init_protein_data_files(self, proteins):
        self.protein_data_files = []
        for input_fp_i in self.input_fp_list:
            print(input_fp_i)
            protein_data_files_i = get_protein_data_files(
                input_fp_i,
                proteins,
                self.re_filter_fn,
                data_type=self.data_type,
                use_enc_scaler=self.use_enc_scaler)
            self.protein_data_files.extend(protein_data_files_i)
    

    def select_proteins(self, ids):
        self.selected_proteins = ids
        self._frames = []
        for frame in self.frames:
            if frame[0] in ids:
                self._frames.append(frame)

    def select_molecules(self, ids):
        return self.select_proteins(ids)

    def deselect_proteins(self):
        self.selected_proteins = None
        self._frames = self.frames

    def deselect_molecules(self):
        self.deselect_proteins()


    def load_data(self, sel_mol=None, sel_traj_fp=None):
        
        # Where actually the MD data is stored.
        self.protein_list = []
        self.frames = []
        self._frames = []

        # Process each protein.
        for prot_data_files_i in self.protein_data_files:
            # Select only one protein.
            if sel_mol is not None and prot_data_files_i.name != sel_mol:
                continue
            self._print("# Loading data for {}".format(prot_data_files_i.name))
            # Parse and get the protein data.
            protein_obj_i = self.load_protein_data(
                prot_data_files=prot_data_files_i, sel_traj_fp=sel_traj_fp)
            self.add_protein_frames(protein_obj_i)
        
        # Store the data attributes with names used in datasets use for
        # other data types.
        self.molecule_list = self.protein_list
        self._frames = self.frames
        if not self._frames:
            raise ValueError("No frames found")

    def _sample_traj_files(self, prot_data_files, sel_traj_fp):
        """Get trajectory files."""
        if sel_traj_fp is None:
            if self.n_trajs is not None:  # Randomly select trajectories.
                traj_fp_l = np.random.choice(prot_data_files.traj_fp_l,
                                                self.n_trajs)
            else:  # Select all trajectories.
                traj_fp_l = prot_data_files.traj_fp_l
        else:  # Select a specific trajectory.
            traj_fp_l = [sel_traj_fp]
        return traj_fp_l


    def add_protein_frames(self, protein_obj):
        """Add the snapshots for this protein to the dataset."""
        n_residues = len(protein_obj.seq)
        protein_count = len(self.protein_list)
        if self.data_type == "xyz":
            protein_data = protein_obj.xyz
        elif self.data_type == "enc":
            protein_data = protein_obj.enc
        else:
            raise KeyError(self.data_type)
        for i in range(protein_data.shape[0]):
            self.frames.append([protein_count, n_residues, i])
        self.protein_list.append(protein_obj)
        
    def refresh_dataset(self):
        if self.n_frames is not None or self.n_trajs:
            self.load_data()


    def get_aa_data(self, prot_idx):
        # Convert to tensors.
        a = self.protein_list[prot_idx].a
        a = torch.tensor(a, dtype=torch.long)
        ae = self.protein_list[prot_idx].ae
        if self.aa_embeddings_dp is not None:
            # Do not convert to tensor, embeddings should be already
            # tensors.
            pass
        data = {"a": a}
        if self.aa_embeddings_dp is not None:
            data["ae"] = ae
        return data
    
    def get_res_ids_data(self, prot_idx):
        if self.protein_list[prot_idx].r is None:
            r = torch.arange(0, len(self.protein_list[prot_idx].seq))
        else:
            r = torch.tensor(self.protein_list[prot_idx].r, dtype=torch.int)
        return {"r": r}
    
    def crop_sequences(self, data, n_residues, idx, use_crops=False):
        if use_crops:
            raise NotImplementedError()
        else:
            crop_data = None
            n_used_residues = n_residues
        return data, crop_data, n_used_residues


################################################################################
# Dataset for xyz cordinates of proteins.                                      #
################################################################################

class ProteinDataset(torch.utils.data.dataset.Dataset,
                     CG_ProteinDatasetMixin):

    def __init__(self,
                 input_fp_list: list,
                 stride: int = 1,
                 n_trajs: int = None,
                 n_frames: int = None,
                 frames_mode: str = "ensemble",
                 accept_prob_rule: Callable = None,
                 proteins: list = None,
                 re_filter_fn: str = None,
                 # input_type: str = "all_atom",
                 input_format: str = "dcd",
                 bead_type: str = "ca",
                 xyz_sigma: float = None,
                 xyz_perturb: dict = None,
                 # aa_embeddings_dp=None,
                 # tbm_mode: str = None,
                 # tbm_lag: int = 1,
                 verbose: bool = False):
        """
        `input_fp_list`: list of paths with input files storing xyz data. Items
            can be filepaths or paths with glob syntax.
        `stride`: stride to use when getting snapshots from the data files.
        `n_trajs`: number of files to randomly select from the entire list of
            files provided with `input_fp_list`. If 'None', all files will be
            used.
        `n_frames`: numer of frames to randomly select from the input data. If
            set to 'None', all frames of the selected trajectories will be used.
        `frames_mode`: determines how to randomly pick frames when `n_frames` is
            not 'None'. Choices: ('trajectory', 'ensemble'). If 'trajectory',
            `n_frames` will be picked from all selected trajectories. If
            'ensemble', `n_frames` will be picked from the total amount of 
            frames (the ensemble) of all trajectories.
        `accept_prob_rule`: TODO.
        `proteins`: list of proteins names to use. If set to 'None', all proteins
            specified in the `input_fp_list` will be used.
        `re_filter_fn`: regular expression to select only some files from
            `input_fp_list`.
        `input_format`: choices are 'numpy' (for numpy files), 'dcd' (for dcd
            trajectory files).
        `xyz_sigma`: standard deviation of the Gaussian noise to add to the xyz
            coordinates. If 'None', no noise will be added.
        `tbm_mode`: template based mode. TODO.
        `tbm_lag`: TODO. Used only if `tbm_mode` is 'lag'.
        `verbose`: use verbose mode.
        """

        self.data_type = "xyz"
        self.use_enc_scaler = False
        if not bead_type in ("ca", "com", "cg"):
            raise KeyError(bead_type)
        self.bead_type = bead_type
        self.xyz_sigma = xyz_sigma
        self.xyz_perturb = xyz_perturb

        self.use_xyz = True
        self.encoder = None

        self._init_common(input_fp_list=input_fp_list,
                          stride=stride,
                          n_trajs=n_trajs,
                          n_frames=n_frames,
                          frames_mode=frames_mode,
                          accept_prob_rule=accept_prob_rule,
                          proteins=proteins,
                          re_filter_fn=re_filter_fn,
                          input_format=input_format,
                          # aa_embeddings_dp=aa_embeddings_dp,
                          tbm_mode=tbm_mode,
                          tbm_lag=tbm_lag,
                          verbose=verbose)
        
        self.load_data()
    
    
    #---------------------------------------------------------------------------
    # Methods for loading the data.                                            -
    #---------------------------------------------------------------------------
    
    def load_top_traj(self, top_fp):
        if top_fp.endswith("pdb"):
            if self.bead_type in ("ca", "com"):
                return self.slice_ca_traj(mdtraj.load(top_fp))
            elif self.bead_type == "cg":
                return self.slice_cg_traj(mdtraj.load(top_fp))

        else:
            raise ValueError("Unknown topology file type: {}".format(top_fp))

    def slice_ca_traj(self, traj):
        ca_ids = [a.index for a in traj.topology.atoms if _check_ca_atom(a)]
        traj = traj.atom_slice(ca_ids)
        return traj

    def slice_cg_traj(self, traj):
        return traj

    def filter_traj(self, traj, is_topology=False):
        if self.bead_type == "ca":
            traj = self.slice_ca_traj(traj)
        elif self.bead_type == "cg":
            traj = self.slice_cg_traj(traj)
        elif self.bead_type == "com":
            traj = slice_traj_to_com(traj, get_xyz=False)
        else:
            raise KeyError(self.bead_type)
        return traj


    def load_protein_data(self, prot_data_files, sel_traj_fp=None):
        """Load data for a single protein."""
        
        self._print("* Loading xyz data")
        
        # Select trajectory files.
        traj_fp_l = self._sample_traj_files(prot_data_files, sel_traj_fp)
        
        # Read xyz data from a trajectory file.
        xyz = []
        top_fp = prot_data_files.top_fp
        # Load the topology.
        top_traj = self.load_top_traj(top_fp)
        seq = "".join([r.code for r in top_traj.topology.residues])
        self._print("+ Sequence: {}".format(seq))

        # Actually parse each trajectory file.
        for traj_fp_i in traj_fp_l:

            self._print("+ Parsing {}".format(traj_fp_i))
            
            # Load the trajectory.
            if self.input_format in ("dcd", ):
                traj_i = mdtraj.load(traj_fp_i, top=top_fp)
                traj_i = self.filter_traj(traj_i)
                xyz_i = traj_i.xyz
            elif self.input_format in ("numpy", ):
                xyz_i = np.load(traj_fp_i)
            else:
                raise KeyError(self.input_format)

            if xyz_i.shape[1] == 0:
                raise ValueError("No atoms found in the parsed trajectory")
                
            self._print("- Parsed a trajectory with shape: {}".format(
                repr(xyz_i.shape)))
            
            # Sample frames with mode "trajectory".
            if self.frames_mode == "trajectory":
                xyz_i = sample_data(data=xyz_i,
                                    n_samples=self.n_frames,
                                    backend="numpy")
                xyz_i = apply_frames_prob_rule(xyz_i, self.accept_prob_rule)
            
            if xyz_i.shape[0] == 0:
                raise ValueError()
                
            # Store the frames.
            self._print("- Selected {} frames".format(repr(xyz_i.shape)))
            xyz.append(xyz_i)

        if not xyz:
            raise ValueError("No data found for {}".format(
                protein_data_files_k.name))
        xyz = np.concatenate(xyz, axis=0)
        
        # Sample frames with mode "ensemble".
        if self.frames_mode == "ensemble":
            xyz = sample_data(data=xyz,
                              n_samples=self.n_frames,
                              backend="numpy")
            xyz = apply_frames_prob_rule(xyz, self.accept_prob_rule)
        self._print("+ Will store {} frames".format(repr(xyz.shape)))

        # Initialize a CG_Protein object.
        protein_obj = CG_Protein(name=prot_data_files.name,
                                 seq=seq,
                                 xyz=xyz,
                                 # aa_embeddings_dp=self.aa_embeddings_dp
                                 )
        return protein_obj

    def __len__(self):
        return len(self._frames)

    def len(self):
        return self.__len__()


    #---------------------------------------------------------------------------
    # Methods for getting the data.                                            -
    #---------------------------------------------------------------------------

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):

        prot_idx, n_residues, frame_idx = self._frames[idx]
        use_crops = False
        data = {}
        
        # xyz data.
        data.update(self.get_xyz_data(prot_idx, frame_idx))
        
        # Amino acid data.
        data.update(self.get_aa_data(prot_idx))

        # Get residue indices.
        data.update(self.get_res_ids_data(prot_idx))
        
        # Crop sequences.
        data, crop_data, n_used_residues = self.crop_sequences(
            data=data, n_residues=n_residues, use_crops=use_crops, idx=idx)
        
        # Additional data, class-dependant (e.g.: encodings).
        data = self._update_graph_args(data, prot_idx, frame_idx, crop_data)

        # Convert into tensors.
        pass

        # Return an object storing data for the selected conformation.
        return StaticData(**data)
    

    def get_xyz_data(self, prot_idx, frame_idx):
        """Returns a xyz frame with shape (L, 3).
        Will also convert to tensors."""
        
        # Get the xyz frame.
        xyz = self.protein_list[prot_idx].xyz[frame_idx]
        # Get a template xyz frame.
        if self.tbm_mode == "random":  # Get a random template frame.
            t_frame_idx = np.random.choice(
                self.protein_list[prot_idx].xyz.shape[0])
            xyz_t = self.protein_list[prot_idx].xyz[t_frame_idx]
        # elif self.tbm_mode == "eval":
        #     raise NotImplementedError()
        elif self.tbm_mode is None:
            xyz_t = None
        else:
            raise KeyError(self.tbm_mode)
        # Perturb with noise.
        if self.xyz_sigma is not None:
            xyz = xyz + np.random.randn(*xyz.shape)*self.xyz_sigma
            if self.tbm_mode is not None:
                xyz_t = xyz_t + np.random.randn(*xyz_t.shape)*self.xyz_sigma
        if self.xyz_perturb is not None and self.xyz_perturb.get("use", True):
            if np.random.rand() < self.xyz_perturb["prob"]:
                if self.xyz_perturb["sched"] == "exp":
                    w_x0 = 1-np.exp(-np.random.rand()*np.e*2)
                    xyz = xyz*w_x0 + np.random.randn(*xyz.shape)*(1-w_x0)
                else:
                    raise KeyError(self.xyz_perturb["sched"])
        # Convert to tensors.
        xyz = torch.tensor(xyz, dtype=torch.float)
        if self.tbm_mode is not None:
            xyz_t = torch.tensor(xyz_t, dtype=torch.float)
        # Return data.
        data = {"x": xyz}
        if self.tbm_mode is not None:
            data["x_t"] = xyz_t
        return data
    
    def _update_graph_args(self, args, prot_idx, frame_idx, crop_data):
        return args


# #---------------------------------------------------------------------------
# # Save an encoded dataset.                                                 -
# #---------------------------------------------------------------------------

# def build_enc_std_scaler(
#     model_params: dict,
#     encoder: Callable,
#     device: str,
#     train_input: list,
#     scaler_fp: str,
#     proteins: list = None,
#     batch_size: int = 32,
#     n_frames: int = None,
#     n_trajs: int = None,
#     frames_mode: str = "ensemble"
#     ):

#     print("\n# Building a standard scaler for the encodings.")

#     # Standard scaler data.
#     encoding_dim = model_params["generative_model"]["encoding_dim"]
#     scaler_e_acc = torch.zeros(encoding_dim)
#     scaler_e2_acc = torch.zeros(encoding_dim)
#     scaler_n_acc = 0

#     # Process xyz data for each molecule.
#     input_files = []
#     for pattern in train_input:
#         for fp in glob.glob(pattern):
#             input_files.append(fp)

#     _input_files = []
#     for prot_idx, prot_data_file_i in enumerate(input_files):
#         if proteins is not None:
#             with open(prot_data_file_i, "r") as i_fh:
#                 prot_data = json.load(i_fh)
#             if prot_data["name"] in proteins:
#                 _input_files.append(prot_data_file_i)
#         else:
#             _input_files.append(prot_data_file_i)

#     for prot_idx, prot_data_file_i in enumerate(_input_files):

#         print("+ Processing %s (%s of %s)." % (
#             os.path.basename(prot_data_file_i), prot_idx+1,
#             len(_input_files)))
        
#         with open(prot_data_file_i, "r") as i_fh:
#             prot_data = json.load(i_fh)

#         # Process all the data to compute the standard scaler statistics.
#         dataset_i = CA_proteinDataset_2(
#             input_fp_list=[prot_data_file_i],
#             stride=1,
#             n_trajs=n_trajs,
#             n_frames=n_frames,
#             frames_mode=frames_mode,
#             input_format=model_params["data"].get("input_format", "dcd"),
#             bead_type=model_params["data"].get("bead_type", "ca"),
#             xyz_sigma=None,
#             aa_embeddings_dp=None,
#             tbm_mode=None,
#             tbm_lag=None,
#             verbose=True)
        
#         encoder_dataloader_i = torch.utils.data.dataloader.DataLoader(
#             dataset=dataset_i,
#             batch_size=batch_size,
#             shuffle=True)

#         processed = 0
#         for batch in encoder_dataloader_i:
#             print("- Collecting data for {}/{}".format(
#                 processed, len(encoder_dataloader_i.dataset)))
#             # Encode the xyz coordinates.
#             batch = batch.to(device)
#             with torch.no_grad():
#                 enc_i = encoder.nn_forward(batch).cpu()
#             # Store data needed to build a standard scaler.
#             scaler_e_acc += enc_i.sum(axis=(0, 1))
#             scaler_e2_acc += torch.square(enc_i).sum(axis=(0, 1))
#             scaler_n_acc += enc_i.shape[0]*enc_i.shape[1]
#             processed += batch.num_graphs

#     # Save encodings transformed with a standard scaler.
#     print("+ Save a standard scaler.")

#     scaler_u = scaler_e_acc/scaler_n_acc
#     scaler_s = torch.sqrt(scaler_e2_acc/scaler_n_acc - torch.square(scaler_u))
#     scaler_u = scaler_u.reshape(1, 1, -1)
#     scaler_s = scaler_s.reshape(1, 1, -1)
#     enc_std_scaler = {"u": scaler_u, "s": scaler_s}
#     torch.save(enc_std_scaler, scaler_fp)

#     return enc_std_scaler


# def save_dataset(model_params,
#                  encoder,
#                  prior_input,
#                  out_dp,
#                  proteins=None,
#                  # partition=None,
#                  device="cpu",
#                  enc_std_scaler=None,
#                  remove_raw_enc=False,
#                  batch_size=32):

#     print("\n# Saving dataset at %s." % out_dp)

#     # Parameters.
#     if model_params["data"].get("prior_crop_size") is not None:
#         raise NotImplementedError()

#     # Setup the dataset directories.
#     if os.path.isdir(out_dp):
#         shutil.rmtree(out_dp)
#     os.makedirs(out_dp)
#     p_out_dp = out_dp  # os.path.join(out_dp, partition)
#     # os.mkdir(p_out_dp)
#     mols_dp = os.path.join(p_out_dp, "molecules")
#     os.mkdir(mols_dp)

#     with open(os.path.join(p_out_dp, "notes.txt"), "w") as o_fh:
#         # o_fh.write("crop_size: {}\n".format(crop_size))
#         pass

#     # Process xyz data for each molecule.
#     input_files = []
#     for pattern in prior_input:
#         for fp in glob.glob(pattern):
#             input_files.append(fp)

#     _input_files = []
#     for prot_idx, prot_data_file_i in enumerate(input_files):
#         if proteins is not None:
#             with open(prot_data_file_i, "r") as i_fh:
#                 prot_data = json.load(i_fh)
#             if prot_data["name"] in proteins:
#                 _input_files.append(prot_data_file_i)
#         else:
#             _input_files.append(prot_data_file_i)

#     for prot_idx, prot_data_file_i in enumerate(_input_files):
        
#         print("+ Processing %s (%s of %s)." % (
#             os.path.basename(prot_data_file_i), prot_idx+1,
#             len(_input_files)))

#         with open(prot_data_file_i, "r") as i_fh:
#             prot_data = json.load(i_fh)
#             # if len(prot_data["chains"]) > 1:
#             #     raise NotImplementedError()
#             prot_name = prot_data["name"]
#             # prot_seq = prot_data["seq"][0]

#         # Actually saves a file with the encodings.
#         # Define the output file path.
#         out_fp = os.path.join(mols_dp, "{}".format(prot_name))
#         out_fp_l = []
#         out_fp_m = {}
#         json_fp = out_fp + ".json.enc"

#         # input_trajectories = []
#         # for pattern in prot_data["trajectories"]:
#         #     for fp in glob.glob(pattern):
#         #         input_trajectories.append(fp)

#         save_dataset_i = CA_proteinDataset_2(
#             input_fp_list=[prot_data_file_i],
#             stride=1,
#             n_trajs=None,
#             n_frames=None,
#             input_format=model_params["data"].get("input_format", "dcd"),
#             bead_type=model_params["data"].get("bead_type", "ca"),
#             xyz_sigma=None,
#             aa_embeddings_dp=None,
#             tbm_mode=None,
#             tbm_lag=None,
#             verbose=True)
        
#         save_dataloader_i = torch.utils.data.dataloader.DataLoader(
#             dataset=save_dataset_i,
#             batch_size=batch_size,
#             shuffle=False)

#         # Save an encoding file for each trajectory file.
#         traj_fp_l = save_dataset_i.protein_data_files[0].traj_fp_l

#         for traj_idx, sel_traj_fp in enumerate(traj_fp_l):
#             # Load data for a single trajectory file.
#             save_dataset_i.load_data(sel_traj_fp=sel_traj_fp)
#             # Encode.
#             enc = []
#             xyz = []
#             saved = 0
#             for batch in save_dataloader_i:
#                 print("- Saving {}/{}".format(
#                     saved, len(save_dataloader_i.dataset)))
#                 # Encode the xyz coordinates.
#                 batch = batch.to(device)
#                 with torch.no_grad():
#                     enc_i = encoder.nn_forward(batch).cpu()
#                 # Store the encodings from the current batch.
#                 enc.append(enc_i)
#                 xyz.append(batch.x.cpu())
#                 saved += batch.num_graphs
#             enc = torch.cat(enc, axis=0)
#             out_fp_i = "{}.{}".format(out_fp, traj_idx)
#             out_fp_l.append(out_fp_i)
#             out_fp_m[out_fp_i] = sel_traj_fp
#             torch.save(enc, out_fp_i + ".enc.pt")
#             # torch.save(torch.cat(xyz, axis=0), out_fp_i + ".xyz.pt")

#             # Save encodings transformed with a standard scaler.
#             if enc_std_scaler is not None:
#                 enc_std = (enc - enc_std_scaler["u"])/enc_std_scaler["s"]
#                 torch.save(enc_std, out_fp_i + ".enc_std.pt")
#                 if remove_raw_enc:
#                     os.remove(out_fp_i + ".enc.pt")
        
#         prot_data["trajectories"] = out_fp_l
#         prot_data["original_trajectories"] = out_fp_m
#         prot_data["seq"] = save_dataset_i.protein_list[0].seq
#         with open(json_fp, "w") as o_fh:
#             o_fh.write(json.dumps(prot_data, indent=2))

        
# #
# # Other... TODO.
# #

# def get_ca_protein_2_dataset(model_params, partition, stage=0,
#                              accept_prob_rule=None):
    
#     s_k = "stage_{}".format(stage)
#     if "train" in model_params[s_k] and model_params[s_k]["train"] is not None:
#         xyz_sigma = model_params[s_k]["train"].get("xyz_sigma")
#         xyz_perturb = model_params[s_k]["train"].get(f"{partition}_xyz_perturb")
#     else:
#         xyz_sigma = None
#         xyz_perturb = None
#     if stage == 0:
#         aa_embeddings_dp = None
#     elif stage == 1:
#         aa_embeddings_dp = model_params["data"].get("prior_aa_embedding_dp")
#     else:
#         raise KeyError(stage)
    
#     dataset = CA_proteinDataset_2(
#         input_fp_list=model_params["data"]["{}_input".format(partition)],
#         stride=1,
#         n_trajs=get_dataset_arg("n_trajs", model_params["data"], stage,
#                                 default_args={0: None}),
#         n_frames=get_dataset_arg("n_frames", model_params["data"], stage,
#                                  default_args={0: None}),
#         frames_mode=get_dataset_arg("frames_mode", model_params["data"], stage,
#                                     default_args={0: "ensemble"}),
#         accept_prob_rule=accept_prob_rule,
#         proteins=get_dataset_arg("{}_proteins".format(partition),
#                                  model_params["data"], stage,
#                                  default_args={0: None}),
#         re_filter_fn=get_dataset_arg("{}_re_filter_fn".format(partition),
#                                      model_params["data"], stage,
#                                      default_args={0: None}),
#         input_format=model_params["data"].get("input_format", "dcd"),
#         bead_type=model_params["data"].get("bead_type", "ca"),
#         # use_geometric=False,
#         xyz_sigma=xyz_sigma,
#         xyz_perturb=xyz_perturb,
#         crop_size=get_dataset_arg("crop_size", model_params["data"], stage,
#                                   default_args={0: None}),
#         aa_embeddings_dp=aa_embeddings_dp,
#         tbm_mode=model_params["generative_model"].get("tbm_mode", None),
#         tbm_lag=model_params["generative_model"].get("tbm_lag", None),
#         verbose=True)

#     return dataset


# def _load_dataset(model_params, batch_size, partition, use_geometric, shuffle):

#     accept_prob_rule = get_accept_prob_rule(model_params, partition)
    
#     dataset = get_ca_protein_2_dataset(
#         model_params=model_params,
#         partition=partition,
#         stage=0,
#         accept_prob_rule=accept_prob_rule)

#     dataloader = get_dataloader(dataset=dataset,
#                                 batch_size=batch_size,
#                                 use_geometric=use_geometric)
    
#     return dataset, dataloader

# def get_dataloader(dataset, batch_size, use_geometric, use_crop_enc=False):
#     if use_geometric:
#         # dataloader = torch_geometric.loader.DataLoader(
#         #     dataset=dataset, batch_size=batch_size, shuffle=shuffle)
#         raise NotImplementedError()
#     else:
#         sampler = Uniform_length_batch_sampler(dataset, batch_size)
#         if not use_crop_enc:
#             dl_cls = torch.utils.data.dataloader.DataLoader
#         else:
#             # dl_cls = CropEncDataloader
#             raise NotImplementedError()
#         dataloader = dl_cls(dataset=dataset, batch_sampler=sampler)
#     return dataloader


# def load_datasets(model_params, batch_size, partitions, stage):
#     """Called in the script for training the autoencoder."""
#     return load_datasets_common(model_params, batch_size, partitions, stage,
#                                 _load_dataset)


class EvalProteinDataset(ProteinDataset):
    """
    Use for evaluation. TODO.
    """
    def __init__(self,
                 name,
                 prot_obj,
                 n_frames=None,
                 xyz_sigma=None,
                 bead_type="ca",
                 # aa_embeddings_dp=None,
                 verbose=True,
                 *args, **kwargs):
        # super(CA_proteinDataset, self).__init__()
        self.n_frames = n_frames
        self.xyz_sigma = xyz_sigma
        self.bead_type = bead_type
        self.xyz_perturb = None
        self.aa_embeddings_dp = None  # aa_embeddings_dp
        self.verbose = verbose

        self.use_xyz = True
        self.tbm_mode = None

        # Where actually the MD data is stored.
        self.protein_list = []
        self.frames = []
        self._frames = []

        # Loads the data.
        protein_count = 0
        if isinstance(prot_obj, mdtraj.Trajectory):
            traj = prot_obj
            seq = "".join([r.code for r in traj.topology.residues])
            all_xyz = traj.xyz
        elif isinstance(prot_obj, str):
            seq = prot_obj
            all_xyz = np.zeros((self.n_frames, len(seq), 3))
        else:
            raise TypeError()
        sel_xyz = sample_data(data=all_xyz, n_samples=self.n_frames)
        n_residues = sel_xyz.shape[1]
        self._print("- Loading xyz=%s for %s" % (
            sel_xyz.shape, name))
        protein = CG_Protein(name=name, seq=seq, xyz=sel_xyz,
                             # aa_embeddings_dp=self.aa_embeddings_dp
                            )
        # Add the snapshots for this IDP to the dataset.
        for i in range(protein.xyz.shape[0]):
            self.frames.append([protein_count, n_residues, i])
        self.protein_list.append(protein)
        
        self.molecule_list = self.protein_list
        self._frames = self.frames
        if not self._frames:
            raise ValueError("No frames found")


################################################################################
# Dataset for encoded structures of proteins.                                  #
################################################################################

class EncodedProteinDataset(torch.utils.data.dataset.Dataset,
                            CG_ProteinDatasetMixin):

    def __init__(self,
                 input_fp_list: list,
                 use_enc_scaler: bool = True,
                 stride: int = 1,
                 n_trajs: int = None,
                 n_frames: int = None,
                 frames_mode: str = "ensemble",
                 accept_prob_rule: Callable = None,
                 proteins: list = None,
                 re_filter_fn: str = None,
                 input_format: str = "torch",
                 # aa_embeddings_dp=None,
                 tbm_mode: str = None,
                 tbm_lag: int = 1,
                 verbose: bool = True
                ):

        self.use_enc_scaler = use_enc_scaler

        self.data_type = "enc"
        self.xyz_perturb = None
        self.use_xyz = False

        self._init_common(input_fp_list=input_fp_list,
                          stride=stride,
                          n_trajs=n_trajs,
                          n_frames=n_frames,
                          frames_mode=frames_mode,
                          accept_prob_rule=accept_prob_rule,
                          proteins=proteins,
                          re_filter_fn=re_filter_fn,
                          input_format=input_format,
                          # aa_embeddings_dp=aa_embeddings_dp,
                          tbm_mode=tbm_mode,
                          tbm_lag=tbm_lag,
                          verbose=verbose)
        
        self.load_data()
    
    
    #---------------------------------------------------------------------------
    # Methods for loading the data.                                            -
    #---------------------------------------------------------------------------

    # def load_protein_data(self, prot_data_files, sel_traj_fp=None):
    #     """Load data for a single protein."""
        
    #     self._print("* Loading enc data")
        
    #     # Select trajectory files.
    #     traj_fp_l = self._sample_traj_files(prot_data_files, sel_traj_fp)

    #     # Read enc data from a trajectory file.
    #     enc = []
    #     # Load the topology.
    #     seq = prot_data_files.seq
    #     self._print("+ Sequence: {}".format(seq))

    #     # Actually parse each trajectory file.
    #     for traj_fp_i in traj_fp_l:

    #         self._print("+ Parsing {}".format(traj_fp_i))
            
    #         # Load the trajectory.
    #         if self.input_format in ("torch", ):
    #             enc_i = torch.load(traj_fp_i)
    #         else:
    #             raise KeyError(self.input_format)
                
    #         self._print("- Parsed a trajectory with shape: {}".format(
    #             repr(enc_i.shape)))
            
    #         # Sample frames with mode "trajectory".
    #         if self.frames_mode == "trajectory":
    #             enc_i = sample_data(data=enc_i,
    #                                 n_samples=self.n_frames,
    #                                 backend="numpy")
    #             enc_i = apply_frames_prob_rule(enc_i, self.accept_prob_rule)
            
    #         if enc_i.shape[0] == 0:
    #             raise ValueError()
                
    #         # Store the frames.
    #         self._print("- Selected {} frames".format(repr(enc_i.shape)))
    #         enc.append(enc_i)

    #     if not enc:
    #         raise ValueError("No data found for {}".format(
    #             protein_data_files_k.name))
    #     enc = np.concatenate(enc, axis=0)
        
    #     # Sample frames with mode "ensemble".
    #     if self.frames_mode == "ensemble":
    #         enc = sample_data(data=enc,
    #                           n_samples=self.n_frames,
    #                           backend="numpy")
    #         enc = apply_frames_prob_rule(enc, self.accept_prob_rule)
    #     self._print("+ Will store {} frames".format(repr(enc.shape)))

    #     # Initialize a CA_Protein object.
    #     protein_obj = CA_Protein(
    #         name=prot_data_files.name,
    #         seq=seq,
    #         xyz=None,
    #         aa_embeddings_dp=self.aa_embeddings_dp)
    #     protein_obj.set_encoding(enc)
    #     return protein_obj

    def __len__(self):
        return len(self._frames)

    def len(self):
        return self.__len__()
    

    #---------------------------------------------------------------------------
    # Methods for getting the data.                                            -
    #---------------------------------------------------------------------------
    
    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):

        prot_idx, n_residues, frame_idx = self._frames[idx]
        n_used_residues = n_residues

        data = {}
        
        # Encoding data.
        data.update(self.get_enc_data(prot_idx, n_residues, frame_idx))
        
        # Amino acid data.
        data.update(self.get_aa_data(prot_idx))

        # Residue indices.
        data.update(self.get_res_ids_data(prot_idx))
        
        # Additional data, class-dependant (e.g.: encodings).
        data = self._update_graph_args(data, prot_idx, frame_idx)

        # Convert into tensors.
        pass

        # Return an object storing data for the selected conformation.
        data_cls = self._get_data_cls()
        return data_cls(**data)
    
    def _get_data_cls(self):
        return StaticDataEnc
    

    def get_enc_data(self, prot_idx, n_residues, frame_idx):
        """Returns a enc frame with shape (L, E_e).
        Will also convert to tensors."""
        
        # Get the enc frame.
        enc = self.protein_list[prot_idx].enc[frame_idx]
        # Get a template enc frame.
        if self.tbm_mode == "random":  # Get a random template frame.
            t_frame_idx = np.random.choice(
                self.protein_list[prot_idx].enc.shape[0])
            enc_t = self.protein_list[prot_idx].enc[t_frame_idx]
        # elif self.tbm_mode == "eval":
        #     raise NotImplementedError()
        elif self.tbm_mode is None:
            enc_t = None
        else:
            raise KeyError(self.tbm_mode)
        # Return data.
        data = {"z": enc}
        if self.tbm_mode is not None:
            data["z_t"] = enc_t
        return data

    
    def _update_graph_args(self, args, prot_idx, frame_idx):
        return args


class EvalEncodedProteinDataset(EncodedProteinDataset):
    """
    Dataset for encodings of a single protein used at inference time for
    generating conformations for that protein.
    """
    def __init__(self,
                 name: str,
                 seq: str,
                 enc_dim: int,
                 n_frames: int,
                 use_enc_scaler: bool = True,
                 # aa_embeddings_dp: str = None,
                 # tbm_mode: str = None,
                 # tbm_lag: int = 1,
                 # input_format: str = "torch",
                 verbose: bool = True
                 ):
        torch.utils.data.dataset.Dataset.__init__(self)
        self.n_frames = n_frames
        self.use_enc_scaler = use_enc_scaler
        self.verbose = verbose
        self.aa_embeddings_dp = None
        self.tbm_mode = None
        self.tbm_lag = None


        self.frames = []
        self._frames = []
        self.protein_list = []

        idx = 0
        n_residues = len(seq)
        enc_i = torch.zeros(self.n_frames, n_residues, enc_dim)
        xyz_i = None
        self._print("- Loading enc=%s for %s" % (enc_i.shape, name))
        protein = CG_Protein(name=name, seq=seq, xyz=xyz_i,
                         # aa_embeddings_dp=self.aa_embeddings_dp,
                        )
        protein.set_encoding(enc_i)
        # Add the snapshots for this IDP to the dataset.
        for i in range(protein.enc.shape[0]):
            self.frames.append([idx, n_residues, i])
        self.protein_list.append(protein)

        self.molecule_list = self.protein_list
        self._frames = self.frames

    def load_custom_xyz_data(self, *args, **kwargs):
        raise TypeError(
            "'load_custom_xyz_data' can not be used in encoded datasets")


# def _load_enc_dataset(model_params, batch_size,
#                       partition, phase, use_geometric, shuffle):

#     crop_size = model_params["data"]["prior_crop_size"]

#     accept_prob_rule = get_accept_prob_rule(model_params, partition, stage=1)

#     #-----------------------------------------------------------------
#     # Build datasets and dataloaders that load encodings from files. -
#     #-----------------------------------------------------------------

#     if crop_size is None:

#         def get_prior_param_legacy(params, partition, partition_pattern, name):
#             partition_name = partition_pattern.format(partition)
#             if partition == "train":
#                 if partition_name in params:
#                     return params[partition_name]
#                 else:
#                     return params[name]
#             elif partition == "val":
#                 return params[partition_name]
#             else:
#                 raise KeyError(partition)

#         dataset = Encoded_CA_proteinDataset_2(
#             input_fp_list=model_params["data"]["prior_{}_input".format(partition)],
#             use_enc_scaler=model_params["generative_model"]["use_enc_std_scaler"],
#             stride=1,
#             n_trajs=get_prior_param_legacy(
#                 params=model_params["data"],
#                 partition=partition,
#                 partition_pattern="prior_{}_n_trajs",
#                 name="prior_n_trajs"),  # model_params["data"]["prior_n_trajs"],
#             n_frames=get_prior_param_legacy(
#                 params=model_params["data"],
#                 partition=partition,
#                 partition_pattern="prior_{}_n_frames",
#                 name="prior_n_frames"),  # model_params["data"]["prior_n_frames"],
#             frames_mode=get_prior_param_legacy(
#                 params=model_params["data"],
#                 partition=partition,
#                 partition_pattern="prior_{}_frames_mode",
#                 name="prior_frames_mode"),  # model_params["data"]["prior_frames_mode"],
#             accept_prob_rule=accept_prob_rule,
#             proteins=model_params["data"]["prior_{}_proteins".format(partition)],
#             re_filter_fn=model_params["data"]["prior_{}_re_filter_fn".format(partition)],
#             input_format=model_params["data"]["prior_input_format"],
#             aa_embeddings_dp=model_params["data"]["prior_aa_embedding_dp"],
#             tbm_mode=model_params["generative_model"].get("tbm_mode"),
#             tbm_lag=model_params["generative_model"].get("tbm_lag"),
#             experimental_dp=model_params["data"].get("prior_experimental_dp"),
#             experimental_features=model_params["generative_model"].get(
#                 "experimental_features"),
#             verbose=True)

#         dataloader = get_dataloader(dataset=dataset,
#                                     batch_size=batch_size,
#                                     use_geometric=False)
    
#     #------------------------------------------------------------------------
#     # Build datasets and dataloaders that convert xyz crops to encodings at -
#     # runtime.                                                              -
#     #------------------------------------------------------------------------

#     else:
        
#         # dataset = get_ca_protein_2_dataset(
#         #     model_params=model_params,
#         #     partition=partition,
#         #     stage=1,
#         #     accept_prob_rule=accept_prob_rule)
#         # dataset.set_encoder(model_params)

#         # dataloader = get_dataloader(dataset=dataset,
#         #                             batch_size=batch_size,
#         #                             use_geometric=False,
#         #                             use_crop_enc=True)
#         raise NotImplementedError()

#     return dataset, dataloader


# def load_enc_datasets(model_params, batch_size, partitions, stage, phase):
#     return load_enc_datasets_common(model_params, batch_size,
#                                     partitions, stage, phase,
#                                     _load_enc_dataset)