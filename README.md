1. Contents
-------------------
This package contains the raw data files as well as source code required for
the analysis. In case of questions or difficulties running the code, please contact
kornfeld@neuro.mpg.de

2. Provided data
-------------------
2.1 Registered raw EM dataset

A streaming configuration file (j0256.conf) is provided for KNOSSOS (www.knossostool.org) that can be used
to easily access all image data and perform your own tracings in the dataset. If the hosting source
changes, an updated streaming configuration file will be provided on github.

The configuration file can also be used together with the knossosDataset class provided
in the public knossos_utils package to directly work with the dataset (e.g. download it, convert
to an image stack, etc).

2.2 Skeleton files

All annotations are provided in a zip file, that contains the necessary data to run the functions of HVC_RA_analysis.py.
The .k.zip files can also be opened directly in KNOSSOS for further analysis. Note that all postsynaptic tracings are contained within
the axon k.zips.

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
