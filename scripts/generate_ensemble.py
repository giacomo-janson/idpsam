import os
import pathlib
import argparse
import re
import numpy as np
from sam.model import SAM
try:
    import cg2all
    has_cg2all = True
except ImportError:
    has_cg2all = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate a conformational ensemble for an input peptide.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config_fp', type=str, required=True,
        help='YAML or JSON configuration file for SAM generative model.')
    parser.add_argument('-s', '--seq', type=str, required=True,
        help='Amino acid sequence of the peptide.')
    parser.add_argument('-o', '--out_path', type=str, required=True,
        help='Output path. File extensions for different file types will be'
            ' automatically added.')
    parser.add_argument('-u', '--out_fmt', type=str, default='dcd',
        choices=['numpy', 'dcd'],
        help='Output format for the file storing xyz coordinates.')
    parser.add_argument('-n', '--n_samples', type=int, default=1000,
        help='Number of samples to generate.')
    parser.add_argument('-t', '--n_steps', type=int, default=100,
        help='Number of diffusion steps (min=1, max=1000).')
    parser.add_argument('-b', '--batch_size', type=int, default=250,
        help='Batch size for sampling.')
    parser.add_argument('--cg2all_batch_size', type=int, default=None,
        help='cg2all batch size for all-atom reconstruction.'
             ' Only takes effect when using the --all_atom option.'
             ' If not provided, it will be equal to --batch_size.')
    parser.add_argument('-a', '--all_atom', action='store_true',
        help='Convert the C-alpha conformations to all-atom via cg2all.')
    parser.add_argument('-d', '--device', type=str, default='cpu',
        choices=['cpu', 'cuda'], help='PyTorch device.')
    parser.add_argument('--cg2all_device', type=str, default='cpu',
        choices=['cpu', 'cuda'],
        help='PyTorch device for cg2all. If CUDA is installed but DGL can not'
            ' find it, use \'cpu\' here.')
    parser.add_argument('-q', '--quiet', action='store_true',
        help='Quiet mode, will not print any output.')
    args = parser.parse_args()


    #---------------
    # Check input. -
    #---------------

    if not re.match(r'^[QWERTYIPASDFGHKLCVNM]*$', args.seq):
        raise ValueError(
            "The input sequence can contain only standard amino acid letters.")

    allowed_aa_out_fmt = ("dcd", )
    if args.all_atom:
        if not has_cg2all:
            raise ImportError(
                "The cg2all library is not installed. Can not reconstruct an"
                " all-atom ensemble. For cg2all installation instructions go"
                " here: https://github.com/huhlim/cg2all")
        if args.out_fmt != "dcd":
            raise ValueError(
                f"The --out_fmt can only be in {repr(allowed_aa_out_fmt)} when"
                " using --all_atom.")
        if args.cg2all_batch_size is None:
            args.cg2all_batch_size = args.batch_size
        if args.n_samples < args.cg2all_batch_size or \
           args.n_samples % args.cg2all_batch_size != 0:
            raise ValueError(
                "The all-atom conversion script can only work when the cg2all"
                f" batch size is an exact divisor of --n_samples. You provided"
                f" {args.batch_size} and {args.n_samples}. Please manually"
                " adjust the values.")
        if args.cg2all_device is None:
            args.cg2all_device = args.device


    #--------------
    # Run idpSAM. -
    #--------------

    # Initialize the idpSAM model.
    model = SAM(config_fp=args.config_fp,
                device=args.device,
                verbose=not args.quiet)

    # Generate C-alpha conformations.
    out = model.sample(seq=args.seq,
                       n_samples=args.n_samples,
                       n_steps=args.n_steps,
                       batch_size_eps=args.batch_size,
                       batch_size_dec=args.batch_size,
                       return_enc=False,
                       out_type="numpy")

    # Save the output data.
    save = model.save(out=out,
                      out_path=args.out_path,
                      out_fmt=args.out_fmt)

    # Optional: reconstruct all-atom details via cg2all.
    if args.all_atom:
        model.cg2all(ca_pdb_fp=save["ca_pdb"],
                     ca_traj_fp=save["ca_dcd"],
                     out_path=args.out_path,
                     batch_size=args.cg2all_batch_size,
                     device=args.cg2all_device)