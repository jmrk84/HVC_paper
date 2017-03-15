1. Contents
-------------------
This package contains the raw data files as well as source code required for
the analysis. In case of questions or difficulties running the code, please contact
kornfeld@neuro.mpg.de

Updates regarding data and code will be made available on
https://github.com/jmrk84/HVC_paper


2. Provided data
-------------------
2.1 Registered raw EM dataset
A streaming configuration file is provided for KNOSSOS (www.knossostool.org) that can be used
to easily access all image data and perform your own tracings in the dataset. If the hosting source
changes, an updated streaming configuration file will be provided on github.

The configuration file can also be used together with the knossosDataset class provided
in the public knossos_utils package to directly work with the dataset (e.g. download it, convert
to an image stack, etc).

2.2 Skeleton files


3. Source code
-------------------
3.1  License
All source code is licensed under the GPL v2 license. 
Copyright, J. Kornfeld, Max Planck Society

3.2 Dependencies

The provided Python source code requires some additional libraries to be installed to run.
* knossos_utils (see https://github.com/knossos-project/knossos_utils for installation
instructions)

* numpy (pip install numpy)
* matplotlib (pip install matplotlib)
* networkx (pip install networkx)
* pandas (pip install pandas)
* mayavi (pip install mayavi)

All pip-installable libraries come with anaconda, it is therefore advised to install the
anaconda Python distribution.
