This directory contains R scripts to perform differential expression analysis
of RNA-seq data using sleuth and a few linear models that group data
differently. The model and grouping are specified at the beginning of each script.

All of the models include batch effects and galactose response through the
'Replicate' and 'condition' coefficients, respectively.

The genotypes are as follows:

Genotype, description
TL47, wild-type
yQC7, truncated RPB1 with 10 CTD heptad repeats
yQC15, truncated RPB1 with 10 CTD heptad repeats fused to FUS LCD
yQC16, truncated RPB1 with 10 CTD heptad repeats fused to TAF15 LCD

Sleuth reference:
Harold J. Pimentel, Nicolas Bray, Suzette Puente, Páll Melsted and Lior Pachter (2017) Nature Methods. Differential analysis of RNA-Seq incorporating quantification uncertainty.
