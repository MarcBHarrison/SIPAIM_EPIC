from epic import Epic
import os
import sys

args = sys.argv[1:]
assert len(args) == 1

image_file = args[0]
# Change this line to the file you want to analyze

base = "/ifs/loni/faculty/thompson/four_d/mharrison/EPIC/DTI_test"
results_dir = "Results_{}".format(os.path.splitext(os.path.split(image_file)[1])[0])
# construct argument dictionary
args = {'info_table_name': os.path.join(base, image_file),
        'num_repeats': 10,
        'num_folds': 2,
        'random_flag': 0,
        'output_dir': os.path.join(base, results_dir),
        'num_lower_repeats': 12,
        'num_lower_folds': 2,
        'classifier_name': "LR",
        'data_type': 'freesurfer',
        'data_type_args': {'first_col': "Thk_L_bankssts_thickavg", 'last_col': "SV_Laccumb"},
        }

# create classification object
epic = Epic(args)

# run epic
epic.merge_optimize()
