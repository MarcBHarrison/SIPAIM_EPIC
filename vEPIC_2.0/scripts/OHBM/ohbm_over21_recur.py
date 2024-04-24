from epic import Epic
import os
import sys

args = sys.argv[1:]
assert len(args) == 1

image_file = args[0]
# Change this line to the file you want to analyze

base = "/ifs/loni/faculty/thompson/four_d/mharrison/OHBM"
results_dir = "Results_{}".format(os.path.splitext(os.path.split(image_file)[1])[0])
# construct argument dictionary
args = {'info_table_name': os.path.join(base, image_file),
        'num_repeats': 10,
        'num_folds': 10,
        'random_flag': 0,
	'merge_flag': 'ClassifierSmallest',
        'output_dir': os.path.join(base, results_dir),
        'num_lower_repeats': 10,
        'num_lower_folds': 2,
        'classifier_name': "Linear SVM",
        'data_type': 'freesurfer',
        'data_type_args': {'first_col': "Thk_L_caudalmiddlefrontal_thickavg", 'last_col': "SV_Ramyg"},
        }

# create classification object
epic = Epic(args)

# run epic
epic.merge_optimize()
