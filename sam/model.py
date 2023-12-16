"""
Module with mostly code for quickly using the SAM model at inference time.
"""

import os
import time
import json
import pathlib
import shutil
import subprocess
import numpy as np
try:
    import yaml
    has_yaml = True
except ImportError:
    has_yaml = False
import torch
from sam.nn.autoencoder.decoder import get_decoder
from sam.nn.noise_prediction.eps import get_eps_network
from sam.diffusion import get_diffusion_model
from sam.data.cg_protein import EvalEncodedProteinDataset, EvalProteinDataset


def read_cfg_file(cfg_fp):
    if cfg_fp.endswith(".json"):
        with open(cfg_fp, "r") as i_fh:
            model_cfg = json.load(i_fh)
    elif cfg_fp.endswith(".yaml"):
        if not has_yaml:
            raise ImportError("Can not read YAML configuration file, the pyyaml"
                              " library is not installed.")
        with open(cfg_fp, 'r') as i_fh:
            model_cfg = yaml.safe_load(i_fh)
    else:
        raise TypeError(
            f"Invalid extension for configuration file: {cfg_fp}. Must be a"
            " json or yaml file.")
    return model_cfg


class SAM:
    """
    Wrapper class for using the SAM models at inference time.
    """

    def __init__(self,
        config_fp: str,
        device: str = "cpu",
        weights_parent_path: str = None,
        verbose: bool = False):

        self.weights_parent_path = weights_parent_path
        self.verbose = verbose

        if self.verbose:
            print(f"# Setting up a SAM model from: {config_fp}.")

        #
        # Initial configuration.
        #

        # Read the configuration file of the model.
        self.model_cfg = read_cfg_file(config_fp)

        # PyTorch device.
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                map_location_args = {}
            else:
                raise OSError("CUDA is not available for PyTorch.")
        elif device == "cpu":
            self.device = torch.device("cpu")
            map_location_args = {"map_location": torch.device('cpu')}
        else:
            raise KeyError(device)

        if self.verbose:
            print("- Using device '%s'." % self.device)

        #
        # Load the epsilon network and the diffusion process object.
        #

        # Network.
        eps_fp = self._get_weights_path(
            self.model_cfg["latent_network"]["weights"])
        if self.verbose:
            print(f"- Loading epsilon network from: {eps_fp}.")
        self.eps_model = get_eps_network(self.model_cfg)
        self.eps_model.load_state_dict(torch.load(eps_fp, **map_location_args))
        self.eps_model.to(self.device)
        self.eps_ema = None

        # Load the standard scaler for the encodings, if necessary.
        if self.model_cfg["generative_model"]["use_enc_std_scaler"]:
            enc_scaler_fp = self._get_weights_path(
                self.model_cfg["generative_model"]["enc_std_scaler_fp"])
            enc_std_scaler = torch.load(enc_scaler_fp)
            enc_std_scaler["u"] = enc_std_scaler["u"].to(dtype=torch.float,
                                                         device=self.device)
            enc_std_scaler["s"] = enc_std_scaler["s"].to(dtype=torch.float,
                                                         device=self.device)
        else:
            enc_std_scaler = None
        self.enc_std_scaler = enc_std_scaler

        # Diffusion process.
        self.diffusion = get_diffusion_model(model_cfg=self.model_cfg,
                                             network=self.eps_model,
                                             ema=self.eps_ema)

        #
        # Load the decoder.
        #

        dec_fp = self._get_weights_path(self.model_cfg["decoder"]["weights"])
        if self.verbose:
            print(f"- Loading decoder network from: {dec_fp}.")
        self.decoder = get_decoder(self.model_cfg)
        self.decoder.load_state_dict( torch.load(dec_fp, **map_location_args))
        self.decoder.to(self.device)
    

    def _get_weights_path(self, path):
        if self.weights_parent_path is None:
            return path
        else:
            b = pathlib.Path(self.weights_parent_path)
            p = pathlib.Path(path)
            return str(b / p)


    def sample(self,
        seq: str,
        n_samples: int = 1000,
        n_steps: int = 100,
        batch_size_eps: int = 256,
        batch_size_dec: int = None,
        prot_name: str = "protein",
        return_enc: bool = False,
        out_type: str = "numpy"):
        """
        Generates a Ca ensemble for a protein chain of sequence `seq`.

        Arguments
        `seq`: amino acid sequence of length L.
        `n_samples`: number of conformations to generate.
        `n_steps`: number of diffusion steps.
        `batch_size_eps`: batch size for sampling with the diffusion model.
        `batch_size_dec`: batch size for the decoding process.
        `prot_name`: name of the input protein sequence.
        `return_enc`: if 'True', returns also the generated encoding.

        Returns
        `out`: a dictionary storing the xyz coordinates in a tensor of shape
            (n_samples, len(seq), 3). If `return_enc` is 'True', also returns
            the generated encodings in a tensor of shape
            (n_samples, len(seq), enc_dim).
        """

        # Generate encodings.
        enc_gen, time_ddpm = self.generate(
            seq=seq,
            n_samples=n_samples,
            n_steps=n_steps,
            batch_size=batch_size_eps,
            prot_name=prot_name,
            return_time=True)

        # Decode to xyz coordinates.
        xyz_gen, time_dec = self.decode(
            enc=enc_gen,
            seq=seq,
            batch_size=batch_size_dec if batch_size_dec is not None \
                       else batch_size_eps,
            prot_name=prot_name,
            return_time=True)

        # Return the output.
        out = {"seq": seq,
               "name": prot_name,
               "xyz": self._to(xyz_gen, out_type),
               "time": {"tot": time_ddpm+time_dec,
                        "ddpm": time_ddpm,
                        "dec": time_dec}}
        if return_enc:
            out["enc"] = self._to(enc_gen, out_type)
        
        return out

    def _to(self, t, out_type):
        if out_type == "numpy":
            return t.cpu().numpy()
        elif out_type == "torch":
            return t.cpu()
        else:
            raise TypeError(type(t))


    def generate(self,
        seq: str,
        n_samples: int = 1000,
        n_steps: int = 100,
        batch_size: int = 256,
        prot_name: str = "protein",
        return_time: bool = False):
        """
        Generate encodings for a protein with amino acid sequence 'seq'.
        """

        if self.verbose:
            print(f"# Generating encodings for {n_samples} samples.")
            print(f"- seq: {seq}")
            print("- Setting up a dataloader for encodings.")

        # Encoding dataset.
        enc_dataset = EvalEncodedProteinDataset(
            name=prot_name,
            seq=seq,
            n_frames=n_samples,
            enc_dim=self.model_cfg["generative_model"]["encoding_dim"],
            verbose=self.verbose)

        # Dataloader.
        enc_dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=enc_dataset, batch_size=batch_size)

        # Actually generate an encoded ensemble.
        if self.verbose:
            print(f"- Generating...")
        encoding_dim = self.model_cfg["generative_model"]["encoding_dim"]
        sample_args = {"n_steps": n_steps}

        tot_graphs = 0
        time_gen = 0
        enc_gen = []

        while tot_graphs < n_samples:
            for batch in enc_dataloader:
                batch = batch.to(self.device)
                time_gen_i = time.time()
                enc_gen_i = self.diffusion.sample(batch, **sample_args)
                time_gen += time.time() - time_gen_i
                # enc_gen_i = enc_gen_i.reshape(-1, n_nodes, encoding_dim)
                enc_gen.append(enc_gen_i)
                tot_graphs += batch.num_graphs
                if self.verbose:
                    print("- Generated %s conformations of %s" % (tot_graphs,
                                                                  n_samples))
                if tot_graphs >= n_samples:
                    break

        if self.verbose:
            print(f"- Done.")

        # Prepare the output encodings.
        enc_gen = torch.cat(enc_gen, axis=0)[:n_samples]
        
        if self.enc_std_scaler is not None:
            enc_gen = enc_gen*self.enc_std_scaler["s"] + self.enc_std_scaler["u"]

        if return_time:
            return enc_gen, time_gen
        else:
            return enc_gen


    def decode(self,
        enc: torch.Tensor,
        seq: str,
        batch_size: int = 256,
        prot_name: str = "protein",
        return_time: bool = False):
        """
        Decode generated encodings into xyz coordinates.
        """

        n_samples = enc.shape[0]

        if self.verbose:
            print(f"# Decoding {n_samples} samples.")
            print("- Setting up a dataloader for xyz conformations.")

        # Dataset for decoding.
        dataset = EvalProteinDataset(
            name=prot_name,
            prot_obj=seq,
            n_frames=n_samples,
            verbose=self.verbose)

        # Dataloader for decoding.
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset, batch_size=batch_size)

        # Actually decode the ensemble.
        time_gen = 0
        tot_graphs = 0
        xyz_gen = []
        if self.verbose:
            print("- Decoding.")
        while tot_graphs < n_samples:
            for batch in dataloader:
                batch = batch.to(self.device)
                batch_y = torch.zeros(batch.x.shape[0],
                                      batch.x.shape[1],
                                      enc.shape[-1],
                                      device=self.device)
                e_gen_i = enc[tot_graphs:tot_graphs+batch.num_graphs]
                n_gen_i = e_gen_i.shape[0]
                pad_gen_batch = n_gen_i <= batch.num_graphs
                if pad_gen_batch:
                    batch_y[:e_gen_i.shape[0]] = e_gen_i
                else:
                    raise NotImplementedError()
                with torch.no_grad():
                    time_gen_i = time.time()
                    xyz_gen_i = self.decoder.nn_forward(batch_y, batch)
                    time_gen += time.time() - time_gen_i
                if pad_gen_batch:
                    xyz_gen_i = xyz_gen_i[:n_gen_i]
                xyz_gen.append(xyz_gen_i)
                tot_graphs += xyz_gen_i.shape[0]
                if self.verbose:
                    print("- Decoded %s graphs of %s" % (tot_graphs, n_samples))
                if tot_graphs >= n_samples:
                    break
        
        if self.verbose:
            print(f"- Done.")

        # Prepare the output xyz coordinates.
        xyz_gen = torch.cat(xyz_gen, axis=0)[:n_samples]

        if return_time:
            return xyz_gen, time_gen
        else:
            return xyz_gen
    

    def save(self,
        out: dict,
        out_path: str,
        out_fmt: str = "dcd"):

        if self.verbose:
            print("# Saving output.")
        out_path = pathlib.Path(out_path)
        
        save_paths = {}

        # Save a FASTA file with the input sequence.
        fasta_path = out_path.parent / (out_path.name + ".seq.fasta")
        save_paths["fasta"] = fasta_path
        if self.verbose:
            print(f"- Saving a FASTA sequence file to: {fasta_path}.")
        with open(fasta_path, "w") as o_fh:
            o_fh.write(f">{out['name']}\n{out['seq']}\n")

        # Save encodings.
        # enc_gen_path = out_path.parent / (out_path.name + ".enc.gen.npy")
        # np.save(enc_gen_path, out["enc"])

        # Save xyz coordinates.
        if out_fmt == "numpy":
            npy_path = out_path.parent / (out_path.name + ".ca.xyz.npy")
            save_paths["ca_npy"] = npy_path
            if self.verbose:
                print(f"- Saving a C-alpha positions npy file to: {npy_path}.")
            np.save(npy_path, out["xyz"])
            
        elif out_fmt == "dcd":
            import mdtraj
            from sam.data.topology import get_ca_topology
            # Get the mdtraj C-alpha topology.
            topology = get_ca_topology(out["seq"])
            # Build a mdtraj C-alpha Trajectory.
            traj = mdtraj.Trajectory(xyz=out["xyz"], topology=topology)
            traj_path = out_path.parent / (out_path.name + ".ca.traj.dcd")
            # Save.
            save_paths["ca_dcd"] = traj_path
            pdb_path = out_path.parent / (out_path.name + ".ca.top.pdb")
            save_paths["ca_pdb"] = pdb_path
            if self.verbose:
                print(f"- Saving a C-alpha trajectory dcd file to: {traj_path}.")
                print(f"- Saving a C-alpha topology PDB file to: {pdb_path}.")
            traj.save(str(traj_path))
            traj[0].save(str(pdb_path))

        else:
            raise KeyError(out_fmt)
        
        return save_paths
    

    def cg2all(self,
        ca_pdb_fp: str,
        ca_traj_fp: str,
        out_path: str,
        batch_size: int = 1,
        device: str = "cpu"
        ):
        # Paths.
        save = {}
        out_path = pathlib.Path(out_path)
        traj_path = out_path.parent / (out_path.name + ".aa.traj.dcd")
        save["aa_dcd"] = traj_path
        top_path = out_path.parent / (out_path.name + ".aa.top.pdb")
        save["aa_top"] = top_path
        # cg2al command.
        cg2all_cmd = [
            "convert_cg2all",
            "-p", str(ca_pdb_fp),
            "-d", str(ca_traj_fp),
            "-o", str(traj_path),
            "--cg", "ca",
            "--device", device,
            "--batch", str(batch_size),
            "-opdb", str(top_path)
        ]
        # Run cg2all.
        if self.verbose:
            print("# Converting to all-atom via the cg2all model.")
        proc = subprocess.run(cg2all_cmd,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise subprocess.SubprocessError(
                f"Error when running cg2all:\n{proc.stderr.decode('utf-8')}")
        # Return results.
        if self.verbose:
            print(f"- Saved an all-atom trajectory dcd file to: {traj_path}.")
            print(f"- Saved an all-atom topology PDB file to: {top_path}.")
        return save