from epic import Epic
import os

base = "/Users/Marc/Documents/Final_csvs/Test/"

for filename in os.listdir(base):
    if os.path.isdir(os.path.join(base, filename)) or filename.startswith("."):
        continue
    for classifier in [
                       "ClassifierSmallest",
                       #"ClassifierLargest"
                      ]:
        f_out, ext = os.path.splitext(filename)
        # construct argument dictionary
        args = {'info_table_name': os.path.join(base, filename),
                'num_repeats': 5,
                'num_folds': 10,
                'random_flag': 0,
                'output_dir': os.path.join(base,
                                           "output_finals",
                                           f_out + classifier),
                'num_lower_repeats': 2,
                'num_lower_folds': 5,
                'classifier_name': "Linear SVM",
                'data_type': 'freesurfer',
                'merge_flag': classifier,
            }

        # create classification object
        epic = Epic(args)

        # run epic
        epic.merge_optimize()
