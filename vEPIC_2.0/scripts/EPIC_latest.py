import os
import pdb
import sys
import traceback

from epic import Epic

# Change this line to the file you want to analyze
# image_file = "EPIC_testing.csv"
image_file = "/ifshome/mharrison/Epic_Tools/test/Over21_Berlin-All_Site_Age_Sex_ICV_Comp.csv"

base = os.path.split(__file__)[0]
results_dir = "epic_test"
# construct argument dictionary
args = {'info_table_name': os.path.join(base, image_file),
        'random_flag': 0,
        'output_dir': os.path.join('/ifshome/briedel/Epic_Tools', results_dir),
        'num_lower_repeats': 5,
        'num_lower_folds': 2,
        'classifier_name': "Linear SVM",
        'data_type': 'freesurfer',
        'data_type_args': {'first_col': "VL_L_Accumbens_area",
                           "last_col": "Thk_R_transversetemporal"},
        }

try:
    # create classification object
    epic = Epic(args)

    # run epic
    epic.run()

except Exception as err:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem()
