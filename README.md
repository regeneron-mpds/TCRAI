TCRAI
-------------

This package provides a series of utilities for the prediction of whether TCRs are able to bind certain peptide:MHC complexes

Subpackages:

 - modelling:
    Provides modules for the predictive modelling, such as sequence processors,
    feature extractors, cross-validation etc
    
 - motif:
    Provides modules for dimensionally reducing the TCRAI fingerprint space, extracting fasta files of CDR3 from collections
    of TCRs, and drawing logos of motifs of collections of CDR3s.
    
 - plotting:
     Provides modules for plotting commonly used figures
     
     
Installation:
-------------

One can locally install via pip, after cloning the repo to your machine/server:

pip install \[-e\] /path/to/tcrai

use the -e flag if you actually want to edit the underlying code of the repo, otherwise this is not needed.

Recommended: 

Place the package in a virtual environment, e.g. conda.
e.g.

conda create -n my_env pip python=3.6

conda activate my_env

\[ check here that `which pip` returns the pip in the conda env, not your global pip \]

pip install /path/to/tcrai


Python Version

Compatibility with python3.6 is known, >3.6 may work, but is not guaranteed.

Runtime
---------

We have provided a notebook: notebooks/simple_binomial_example.ipynb; which shows a simple example of how to use the TCRAI framework for a single case. It shows how to build TCRAI models, train them, view motifs of identified clusters, and save/load models. This notebook can be easily adapted to datasets with different naming conventions, altering the number and types of inputs and so on. We expect that for most users, adaption of this notebook will be sufficient to allow them to study new datasets and hypotheses. 

Should one wish to use different types of encoders for the CDR3s or genes, one can build new functions in the tcrai/modelling/extractors.py module following some simple rules such that they will "plug-and-play" with the rest of the TCRAI framework. Similarly with the processing and final stages of the model. 

A notebook: notebooks/gene_usage.ipynb; is provided showing how to construct Sankey diagrams of gene usage for a set of TCRs. This can be used with the clusters found in our manuscript, or other fasta files of CDR3 sequences. 


Scripts used to generate manuscript results
-----------------------------------------------

Several python scripts are provided that will train and save models specific to the manuscript, though they could also be adapted to different problems.

The scripts used to generate figure in the paper are collected in scripts/, they are designed to work with the csv data used for this manuscript. The scripts are command line tools, descriptions of the runtime options and inputs for command line use can be found using python /path/to/script.py --help, or alternatively they are provided in scripts/help.txt.

We have done our best to ensure cross-system reproducibility across systems by setting the random seeds of the various libraries used in the code to specific values. It may be possible that there remain some cross-system differences, particularly for GPU v CPU differences.

License
--------

License can be found in the file LICENSE.