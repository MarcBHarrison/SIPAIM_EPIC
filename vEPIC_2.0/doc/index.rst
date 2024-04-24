####
EPIC
####

.. contents::

*************************
Installation Instructions
*************************

Install requires a python3 environment. I recommend using anaconda for linux, once
anaconda is installed, I recommend making an environment to use for epic
analysis, and ipython3 to use debugging capabilities::

   conda update conda
   conda install pip
   conda create --name epic_analysis python=3
   conda install ipython3
   source activate epic_analysis

Once conda is installed, create your local epic library by checking out epic
from Marc's ifshome::

   git clone /ifshome/mharrison/Epic_Tools/ my_epic_branch

Make a branch to do your local development. Replace ``my_epic_branch``
with a meaningful name for your analysis project::

   cd my_epic_tools
   git checkout -b my_epic_branch

Install epic tools dependencies::

   pip install -r requirements.txt
*************************
UPDATING your version to epic2.0
*************************
cd to your epic dir
git remote -v # look and see what the name of Marc's epic is -- probably called origin
git branch # look and see if you have an epic2.0 branch already.
If so:
git checkout epic2.0
If not:
git checkout -b epic2.0
git pull origin epic2.0 # replace origin with the appropriate name from the remote -v call


*************************
*************************
*************************
NOTE: Most of this below is for the old version of EPIC
*************************
*************************
*************************
Parameters
*************************
In your_script_name.py you need to set: the number of folds, classifier type,
filepath to your csv, first/last column names of your csv

In your data.py in the "epic" dir, you need to set: prefixes of all variable classes on line 132
This is on the 'assert self.prefixes.issubset' line - if your feature type has the prefix 'VL_ROI' and
'Thk_ROI' then you would add 'VL' and 'Thk' - not ID or Group
Also set the SVM baseline cutoff threshold 'cutoff_threshold' - you can also set this to 0

In your epic.py in the "epic" dir, you need to set: the c statistic to optimize your results
This is on the 'clf = svm.LinearSVC' line

In your freesurfer.py script (or whatever it's called) there is a QC parameter in epic.py line 308 onwards
That will prevent you from being able to run with more folds than you have data/group size for
YOU NEED TO HAVE AT LEAST 3 DX/CN PEOPLE IN EACH FOLD! If not, reduce folds!

*************************
Running Instructions
*************************
Make a script in the scripts folder, you can use one of the existing
ones as a template. If you are using a conda environment, you can't just
include ``#!/usr/bin/env python`` as the first line to run your script.
Instead, activate your python 3 environment, and then run::

   source activate epic_analysis
   ipython3 --pdb
   %run filepath/scripts/your_script_name.py
This will make sure to use the local version of python that's located in your
working python environment.

When you are finished running analyses, to return to the normal environment,
hit control-d enter and then

   source deactivate epic_analysis

=========
Debugging
=========
If/when it reaches an uncaught exception ipython3 will jump into debug mode.
If/when that happens, here's the basics you need
to know in order to figure out what went wrong.

* You can list out the surrounding lines of code if you type ``l``
* You can travel up the call stack by typing ``u``
* You can travel back down the call stack by typing ``d``
* You can list out values of variables that are located at your current level
  of the call stack by typing the variable name.
* You quit the debugger by typing ``q``

Those are about the only commands that I use. If you want to learn more,
check out the `pdb docs`_.

.. _pdb docs: https://docs.python.org/3.5/library/pdb.html


*********************************************
Saving Your Work and Sharing it with your Lab
*********************************************

Make sure to run::

   git commit -a

Every once in a while in order to version control your work. If you make some
 interesting changes to epic tools and you want to share, push your branch
 back to Marc's repository so she can merge your changes with the master
 branch. To do this, run the following command after running git commit::

   git push origin my_epic_branch


********
Misc
********

If you want to run multiple verions or csvs you can call them in separate terminal
tabs after activating epic_tools and opening ipython3 with (you can do this in 1 tab,
but they won't run in parallel):

%run Epic_Tools/scripts/freesurfer_analysis2.py /ifs/loni/faculty/thompson/four_d/
   mharrison/ABIDE/MDD_ISBI/MDD_full_matched_21.csv
%run Epic_Tools/scripts/freesurfer_analysis3.py /ifs/loni/faculty/thompson/four_d/
   mharrison/ABIDE/MDD_ISBI/MDD_full_matched_21.csv

.. toctree:: Contents

   about_epic
   epic_v2

********
Updates
********
Added squared-hinge loss function to SVM
-- this will make SVM more sensitive to large outliers
Added random state to ensure consistency
