# IdpSAM: latent diffusion model for protein conformation generation

## About
Repository implementing [idpSAM](https://todo.com) in [PyTorch](https://pytorch.org). IpdSAM is a latent [diffusion model](https://en.wikipedia.org/wiki/Diffusion_model) for generating C-alpha conformations of [intrinsically disordered proteins](https://en.wikipedia.org/wiki/Intrinsically_disordered_proteins) (IDPs) and peptides. The model was trained on a dataset of Markov Chain Monte Carlo simulations of 3,259 intrinsically disordered regions. The sequences of the peptides were obtained from the [DisProt](https://www.disprot.org) database. The simulations were carried out using [ABSINTH](https://pubmed.ncbi.nlm.nih.gov/18506808/), an [implicit solvent model](https://en.wikipedia.org/wiki/Implicit_solvation), implemented in the [CAMPARI 4.0](https://campari.sourceforge.net/V4/index.html) package. Here we provide code and weights of a pre-trained idpSAM model.

## Applications
This repository can be used for the following applications (see below for more information):
* Generate C-alpha ensembles with a pre-trained idpSAM model.
* Generate all-atom ensembles with a pre-trained idpSAM model and the [cg2all model](https://github.com/huhlim/cg2all) for all-atom reconstruction.
* Train a SAM model on your own dataset of protein conformations.

# Install
## Local system
We recommend to install and run this package in a new [Conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that you create from the `sam.yml` file in this repository. If you follow this strategy, use these commands:

1. Clone the repository:
   ```bash
   git clone https://github.com/giacomo-janson/idpsam.git
   ```
   and go into the root directory of the repository.
2. Install the dedicated conda environment and dependencies:
   ```bash
   conda env create -f sam.yml
   ```
3. Activate the environment:
   ```bash
   conda activate sam
   ```
4. Install the `sam` Python library in editable mode (it will just put the library in [$PYTHONPATH](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH)):
   ```bash
   pip install -e .
   ```
## Run on the cloud
If you want to use idpSAM on the cloud (no installations needed on your system) we have a [Colab notebook](colab).

# Usage
## Generate C-alpha ensembles
### Running locally
TODO
### Running remotely
TODO
## Generate all-atom ensembles
### Running locally
TODO
### Running remotely
TODO

## Train a new SAM model with a custom C-alpha dataset
Using the scripts in the `scripts/training` directory of this repository you can train (on your local system) a SAM model on your own dataset of protein conformations. For more information and requirements, follow the guide at `scripts/training/README.md`.

# Updates
TODO.

# References
TODO.