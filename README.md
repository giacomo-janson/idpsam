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
5. Optional, only if you want to perform all-atom reconstruction when using the idpSAM inference script. Install the [cg2all package](https://github.com/huhlim/cg2all):
   ```bash
   pip install git+http://github.com/huhlim/cg2all
   ```
   this is the command for performing a [CPU-only installation](https://github.com/huhlim/cg2all/#installation) of cg2all. You can also attempt the GPU installation, which involves more steps. If you can't install cg2all with GPU support, the CPU installation is still good for idpSAM. This is because for short peptides cg2all is reasonably fast when running on a CPU. This is the setup we typically use for in-house studies.
## Run on the cloud
If you want to use idpSAM on the cloud (no installations needed on your system) we have a [idpSAM Colab notebook](https://colab.research.google.com/github/giacomo-janson/idpsam/blob/main/notebooks/idpsam_experiments.ipynb).

# Usage
## Generate conformational ensembles
### Running locally
You can generate a structural ensemble of a custom peptide sequence via the  `scripts/generate_ensemble.py` inference script. Its usage is:
```bash
python scripts/generate_ensemble.py -c config/models.yaml -s MFDNASTRNNKRERGKRQGKQTRTQRHADRSQT -o peptide -n 1000 -a -d cuda
```
Here is a description of the arguments:
* `-c`: configuration file for idpSAM. Use the default one provided in the `config` directory of the repository.
* `-s`: amino acid sequence of the instrinsically disordered peptide that you want to model.
* `-o`: output path. In this example, the command will save a series of files named `peptide.*` containing the output. These are DCD trajectory files storing the conformations you generated and PDB files storing that you can use as topologies for parsing the DCD files. Files with the `ca` code store only C-alpha atoms (this is the original output of idpSAM), files with the `aa` code store all-atoms conformations reconstructed by the [cg2all model](https://github.com/huhlim/cg2all) as a post-processing step.
* `-n`: number of conformations to generate.
* `-a`: flag for using cg2all to reconstruct all-atom details from C-alpha traces. You must first install the cg2all package to use this option. 
* `-d`: PyTorch device for the idpSAM models. If you want to generate ensembles with large number of conformations, we strongly reccommend to use GPU support, via the `cuda` value here. By default, the cg2all mode will run on CPU, since it is still fast.

There are also other options that you can tweak. Use the `--help` flag to get the full list of list of them. 
### Running remotely
You can easily generate a C-alpha (and optionally all-atom) ensemble for a custom peptide using a [Colab notebook on the cloud]((https://research.google.com/colaboratory/)). Once you have generated the ensemble, you can download it to your local system. The output will consists of DCD trajectory files, that you can parse with [MDTraj](https://github.com/mdtraj/mdtraj) for example. If you plan to generate large ensembles (> 1000 conformations), it will probably take a really long time (hours) if you are using a CPU runtime. If possible, use a GPU runtime to greatly accelerate idpSAM (few minutes).

Launch the notebook using the link below:

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giacomo-janson/idpsam/blob/main/notebooks/idpsam_experiments.ipynb)


## Train a new SAM model with a custom C-alpha dataset
Using the scripts in the `scripts/training` directory of this repository you can train (on your local system) a SAM model on your own dataset of protein conformations. For more information and requirements, follow the guide at `scripts/training/README.md`.

# Updates
TODO.

# References
TODO.