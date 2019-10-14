# RNA Pol II Length and Disorder Enable Cooperative Scaling of Transcriptional Bursting
This repository contains all the code used for data analysis in the paper above by Porfirio Quintero-Cadena, Tineke L. Lenstra and Paul W. Sternberg.

## analysis
Generate data that can be visualized: extract features from imaging, sequencing and simulation data.
## figures
Generate all main and supplementary figures. Every figure\*, can be generated and saved in `figures/output` by running `python figures/figureX.py` from this directory, where X is the figure number.
\*Except for Figure S2, which requires a pickled classifier whose size is larger than github's file size limit.
## utils
Supporting functions for data analysis.
## data
Output from `analysis` functions used to generate figures. Mostly `csv` files
