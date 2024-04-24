from epic import Epic
import os

# Change this line to the file you want to analyze
image_file = "Dublin_Sydney_Houston.csv"

base = "/ifs/loni/faculty/thompson/four_d/mharrison/MDD/Complete/Single_Sites_Only/Added_To_Single"
results_dir = "Results_mharrison_test_{}".format(os.path.splitext(image_file)[0])
# construct argument dictionary
args = {'info_table_name': os.path.join(base, image_file),
        'num_repeats': 10,
        'num_folds': 5,
        'random_flag': 0,
        'output_dir': os.path.join(base, results_dir),
        'num_lower_repeats': 10,
        'num_lower_folds': 2,
        'classifier_name': "Linear SVM",
        'merge_flag': "ClassifierSmallest",
        'data_type': 'freesurfer',
        'data_type_args': {'first_col': "VL_L_Accumbens_area", 'last_col': "Thk_R_transversetemporal"},
        }

# create classification object
epic = Epic(args)

# run epic
epic.merge_optimize()
