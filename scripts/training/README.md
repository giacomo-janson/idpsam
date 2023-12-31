# Train a custom SAM model
Here you will find minimal scripts to train a new SAM model on a custom protein conformation dataset.

In this example in tutorial form, a simple dataset will be downloaded and used to train a SAM model from scratch. You can probably easily adapt the scripts to your own data (see [below](using-a-custom-dataset)).

# Overview
To train and evaluate a model, run the following scripts in a sequential way.

1. `setup_datasets.py`: download and setup the training and validation sets.
2. `train_ae.py`: train an E(3)-invariant autoencoder (AE).
3. `eval_ae.py`: evaluate the AE (optional).
4. `save_encodings.py`: encode the training set.
5. `train_ddpm.py`: train the DDPM on the encoded training set.
6. `eval_ddpm.py`: evaluate the SAM model you trained.

For each script, you will find more details below.

# Stage 1: `setup_datasets.py`
Run this script to download a dataset of protein conformations. The dataset consists of 10 MCMC trajectories (9 training, 1 validation) for 3 intrinsically disordered peptides from the test set of the [idpSAM article](TODO).

Use the following command to run the script:

```bash
python scripts/training/setup_dataset.py -c config/training/example.yaml
```

where `-c` specifies the configuration file used throughout this whole tutorial. You can find the `example.yaml` file in this repository.

In this step, the only option you might want to change in the configuration file is the `data -> out_dp` key. This is the directory where the dataset will be downloaded and where the remaning scripts will read/write most data. You can use any custom location.

# Stage 2: `train_ae.py`
The script will train a E(3)-invariant AE on the dataset.

```bash
python scripts/training/train_ae.py -c config/training/example.yaml
```

By default this script (and others below) will use `cuda` as a PyTorch device, but you can change this in the configuration file. Training the AE should take XXX with cuda support with this dataset!

The hyper-parameters of the AE are the same of the default idpSAM model. You don't need to change them in this simple experiment. If training on different data, you might need to change them in the `example.yaml` configuration file to obtain good reconstruction performance on your data (see below).

# Stage 3: `eval_ae.py`
The script will run the AE on validation data.

```bash
python scripts/training/train_ae.py -c config/training/example.yaml
```

The goal here is to evaluate the reconstruction ability of the AE.

If you used all the default settings of this tutorial (dataset, hyper-parameters) the performance will likely be good, so you don't need to worry a lot about this step.

You should probably pay attention to this step if using your own dataset. When evaluating on validation data, you should obtain a [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) (PCC) between original and reconstructed Cα-Cα distance of at least 0.99 before training a DDPM. In our experience, a lower PCC will result in reconstruction capability being likely too low for training a good latent diffusion model.

This script will also save some trajectories of the reconstructued 3D structures in case you want to inspect them.

# Stage 4: `save_encodings.py`
Run this script to convert the training set 3D structures into encodings. The encodings will be saved in PyTorch tensor files and will be used to train the diffusion model in the step below.

```bash
python scripts/training/save_encodings.py -c config/training/example.yaml
```

# Stage 5: `train_ddpm.py`

```bash
python scripts/training/train_ddpm.py -c config/training/example.yaml
```

# Stage 6: `eval_ddpm.py`

```bash
python scripts/training/eval_ddpm.py -c config/training/example.yaml
```

This script will also save some trajectories of the generated 3D structures in case you want to inspect them.

# Using a custom dataset
Ok.