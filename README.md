# AisNet

##### Requirements:
- python 3
- ASE
- numpy
- PyTorch (>=0.4.1)
- h5py
- schnetpack (3.19)

_**Note: We recommend using a GPU for training the neural networks.**_

## Installation

#### Install requirements

```
cd ./pack/schnetpack
pip install -r requirements.txt
```

#### Install Pack

```
pip install .
```

## Getting started

The example scripts had been provided in ./script

### Cu example

First train the Encoding module, please call:

```
cd ./script
python Cu_encoding_train.py
```

Put the folder obtained after training in the same directory and continue to call:

```
python Cu_example.py
```

The full AisNet will be trained.