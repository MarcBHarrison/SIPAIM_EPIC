##########
About EPIC
##########

.. contents::



*************
Boruta Method
*************

From https://m2.icm.edu.pl/boruta/

| The core idea is that a feature that is relevant is more
| useful for classification than its version with a permuted order
| of values. To this end, Boruta extends the given dataset with such
| permuted copies of all features (which we call shadows), applies
| some feature importance measure (aka VIM; Boruta can use any, the
| default is Random Forest's MDA) and checks which features are
| better and which are worse than a most important shadow.


Add an input variable that defines what type of data we are
working with. Options are only freesurfer for the time being.

Merge functions that should be defined:

A freesurfer merge:
   Make freesurfer merge do the following: Given as an input an
   ordered list of all variables, where ordering is according to
   classifier contribution, return a new set of variables that
   merges the lowest ranked classifier with the next lowest ranked
   classifier OF THE SAME DATA TYPE. Merge means that each subject
   receives a new variable who's contents are the average of the
   contents of the previous two variables.

A connectomics merge:
   For connectomics, merging a region means treating two separate
   ROI's within the image as one super-region. Conversely, Freesurfer
   parcellations are numerically averaged.

Ensure a log is kept that documents the new variable sets used and
defines the meanings of any merged variables. Then, run again
based off the returned variable selection.

Make sure that the final as-used dataset is saved with the output.
