from epic import Epic
import os

# TODO, This only runs one classifier right now

base = "/ifs/loni/faculty/thompson/four_d/mharrison/Machine_Learning_MDD/"

# construct argument dictionary
args = {'info_table_name': os.path.join(base, "txt/EPIC_MDD.csv"),
        'num_repeats': 10,
        'num_folds': 10,
        'random_flag': 0,
        'output_dir': os.path.join(base, "results/MDD_SVM"),
        'num_lower_repeats': 10,
        'num_lower_folds': 10,
        'classifier_name': "Linear SVM",
#        'data_type': 'freesurfer',
#        'data_type_args': {'first_col': "VL_asdf",
#                           'last_col': "CT_fdsa"},
        }

# create classification object
epic = Epic(args)

# run epic
epic.merge_optimize()

