import os
import json
import pathlib
import argparse
import time
import numpy as np
import torch

from sam.nn.autoencoder.decoder import get_decoder
from sam.nn.noise_prediction.eps import get_eps_network
from sam.diffusion import get_diffusion_model
from sam.data.cg_protein import EvalEncodedProteinDataset, EvalProteinDataset


parser = argparse.ArgumentParser(
    description='Generate a C-alpha ensemble for a user-defined peptide.')
parser.add_argument('-c', '--config_fp', type=str, required=True,
    help='JSON configuration file for SAM generative model.')
parser.add_argument('-s', '--seq', type=str, required=True,
    help='Amino acid sequence of the peptide.')
parser.add_argument('-o', '--out_path', type=str, required=True,
    help='Path for the output directory. Will save output files inside.')
parser.add_argument('-u', '--out_fmt', type=str, default='numpy',
    choices=['numpy', 'dcd'],
    help='Output format for the file storing xyz coordinates.')
parser.add_argument('-n', '--n_samples', type=int, default=10000,
    help='Number of samples to generate.')
parser.add_argument('-t', '--n_steps', type=int, default=100,
    help='Number of diffusion steps (default=100, min=1, max=1000).')
parser.add_argument('-b', '--batch_size', type=int, default=256,
    help='Batch size for sampling.')
parser.add_argument('-d', '--device', type=str, default='cpu',
    choices=['cpu', 'cuda'], help='PyTorch device.')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()


#
# Initial configuration.
#

# Read the configuration file of the model.
with open(args.config_fp, "r") as i_fh:
    model_cfg = json.load(i_fh)

# PyTorch device.
if args.device == "cuda":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise OSError("CUDA is not available for PyTorch.")
elif args.device == "cpu":
    device = torch.device("cpu")
else:
    raise KeyError(args.device)

if args.verbose:
    print("# Using device '%s'." % device)
if torch.cuda.is_available():
    map_location_args = {}
else:
    map_location_args = {"map_location": torch.device('cpu')}

# Protein data.
prot_name = "eval_protein"

#
# Load the epsilon network and the diffusion process object.
#

# Network.
if args.verbose:
    print("# Loading epsilon network.")
eps_model = get_eps_network(model_cfg)
eps_model.load_state_dict(torch.load(model_cfg["latent_network"]["weights"],
                                     **map_location_args))
eps_model.to(device)
eps_ema = None

# Load the standard scaler for the encodings, if necessary.
if model_cfg["generative_model"]["use_enc_std_scaler"]:
    enc_std_scaler = torch.load(
        model_cfg["generative_model"]["enc_std_scaler_fp"])
    enc_std_scaler["u"] = enc_std_scaler["u"].to(dtype=torch.float,
                                                 device=device)
    enc_std_scaler["s"] = enc_std_scaler["s"].to(dtype=torch.float,
                                                 device=device)
else:
    enc_std_scaler = None

# Diffusion process.
diffusion = get_diffusion_model(model_cfg=model_cfg,
                                network=eps_model,
                                ema=eps_ema)

#
# Load the decoder.
#

if args.verbose:
    print("# Loading decoder network.")
decoder = get_decoder(model_cfg)
decoder.load_state_dict(torch.load(model_cfg["decoder"]["weights"],
                                   **map_location_args))
decoder.to(device)


#
# Generate the encodings.
#

if args.verbose:
    print("# Setting up a dataloader for encodings.")
# Encoding dataset.
enc_dataset = EvalEncodedProteinDataset(
    name=prot_name,
    seq=args.seq,
    n_frames=args.n_samples,
    enc_dim=model_cfg["generative_model"]["encoding_dim"],
    verbose=args.verbose)

# Dataloader.
enc_dataloader = torch.utils.data.dataloader.DataLoader(
    dataset=enc_dataset, batch_size=args.batch_size)

# Actually generate an encoded ensemble.
n_nodes = len(args.seq)
encoding_dim = model_cfg["generative_model"]["encoding_dim"]
sample_args = {"n_steps": args.n_steps}

tot_graphs = 0
time_gen = 0
enc_gen = []
if args.verbose:
    print("# Generating.")
    print(f"- seq: {args.seq}")

while tot_graphs < args.n_samples:
    for batch in enc_dataloader:
        batch = batch.to(device)
        time_gen_i = time.time()
        enc_gen_i = diffusion.sample(batch, **sample_args)
        time_gen += time.time() - time_gen_i
        enc_gen_i = enc_gen_i.reshape(-1, n_nodes, encoding_dim)
        enc_gen.append(enc_gen_i)
        tot_graphs += batch.num_graphs
        if args.verbose:
            print("- Generated %s conformations of %s" % (tot_graphs,
                                                          args.n_samples))
        if tot_graphs >= args.n_samples:
            break

enc_gen = torch.cat(enc_gen, axis=0)[:args.n_samples]

#
# Decode the generated encodings.
#

# Prepare the input encodings.
if enc_std_scaler is not None:
    enc_gen = enc_gen*enc_std_scaler["s"] + enc_std_scaler["u"]

if args.verbose:
    print("# Setting up a dataloader for xyz conformations.")

# Dataset for decoding.
dataset = EvalProteinDataset(
    name=prot_name,
    prot_obj=args.seq,
    n_frames=args.n_samples,
    verbose=args.verbose)

# Dataloader for decoding.
dataloader = torch.utils.data.dataloader.DataLoader(
    dataset=dataset, batch_size=args.batch_size)

# Actually decode the ensemble.
tot_graphs = 0
xyz_gen = []
if args.verbose:
    print("# Decoding.")
while tot_graphs < args.n_samples:
    for batch in dataloader:
        batch = batch.to(device)
        batch_y = torch.zeros(batch.x.shape[0],
                              batch.x.shape[1],
                              enc_gen.shape[-1],
                              device=device)
        e_gen_i = enc_gen[tot_graphs:tot_graphs+batch.num_graphs]
        n_gen_i = e_gen_i.shape[0]
        pad_gen_batch = n_gen_i <= batch.num_graphs  # n_gen_i < batch.num_graphs
        if pad_gen_batch:
            batch_y[:e_gen_i.shape[0]] = e_gen_i
        else:
            raise NotImplementedError()
        with torch.no_grad():
            time_gen_i = time.time()
            xyz_gen_i = decoder.nn_forward(batch_y, batch)
            time_gen += time.time() - time_gen_i
        xyz_gen_i = xyz_gen_i.cpu().numpy()
        if pad_gen_batch:
            xyz_gen_i = xyz_gen_i[:n_gen_i]
        xyz_gen.append(xyz_gen_i)
        tot_graphs += xyz_gen_i.shape[0]
        if args.verbose:
            print("- Decoded %s graphs of %s" % (tot_graphs, args.n_samples))
        if tot_graphs >= args.n_samples:
            break

xyz_gen = np.concatenate(xyz_gen, axis=0)[:args.n_samples]

#
# Save the data.
#

out_path = pathlib.Path(args.out_path)
if not out_path.is_dir():
    os.mkdir(out_path)

enc_gen_path = out_path / "enc.gen.npy"
np.save(enc_gen_path, enc_gen.cpu().numpy())

if args.out_fmt == "numpy":
    np.save(out_path / "xyz.gen.npy", xyz_gen)
    
elif args.out_fmt == "dcd":
    import mdtraj
    from sam.data.topology import get_ca_topology

    topology = get_ca_topology(args.seq)
    traj = mdtraj.Trajectory(xyz=xyz_gen, topology=topology)
    traj.save(str(out_path / "traj.gen.dcd"))
    traj[0].save(str(out_path / "top.pdb"))

else:
    raise KeyError(args.out_fmt)

with open(out_path / "seq.fasta", "w") as o_fh:
    o_fh.write(f">{prot_name}\n{args.seq}\n")