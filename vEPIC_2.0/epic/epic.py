"""
Epic primary python module -- contains the Epic class.
"""

import copy
from io import StringIO
import os
import sys

from boruta import BorutaPy
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import svm

from epic.data import EpicDataSource

class Capturing(list):
    """
    Used to capture stdout from boruta calls and pipe to the Epic log
    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout


class Epic:
    """
    Implements the evolving partitions to improve connectomics (epic)
    algorithm by Gautam Prasad. See doc/about_epic.rst for details on the
    algorithm implementation.
    """

    def __init__(self, args):
        """
        Initialize Epic by inputting the desired run parameters within the args
        dictionary. Expected parameters are as follows:

        info_table_name: A CSV with your covariates and file locations
           for your image files. No default setting.

           NOTE: Ensure your data is relatively proportional between
           your controls and analysis subjects. Therefore, make sure that:

              num_folds must be < 3x Group==1

           Otherwise, the classifiers will overfit each fold based off
           the control group characteristics.

        num_folds: Number of iterations to run given the number of
           cross-validations.

        random_flag: Passed into scikit-learn stratifiedKFold:
           Populates the pseudo random number generators initial state

        output_dir: The directory where the log file and output files
            will be saved. Defaults to the current working directory.

        num_lower_repeats: The number of times to create subfold-groups for a
           partition_features data set. Partioned_features data will be
           classified ``num_lower_repeats * num_lower_folds`` times, then the
           average classification performance is taken from those runs. See
           score_partition and evaluate_classifier for details.

        num_lower_folds: The number of fold groups to split a
           partitioned_features data set into. See num_lower_repeats for what
           this is used for.

        data_type: The data structure type used to load EPIC. Options are
           defined within epic.data. Includes freesurfer and connectomics.

        data_type_args: a dictionary of arguments to pass to the data
           interface specified by data_type. See epic.data classes for details.

        classifier_name: The scikit-learn machine learning algorithm
           used to classify the data. Options include:

           'Linear SVM', 'RBF SVM', 'LDA', or 'LR'

        dont_log: If true, print output to screen, if false, print
           output to a log file.
        """

        self.info_table_name = args['info_table_name']
        if not os.path.exists(self.info_table_name):
            raise ValueError("Cannot load info table, '{}' is not a valid"
                             " filepath!".format(args['info_table_name']))

        self.num_repeats = int(args.get('num_repeats', 1))
        self.num_folds = int(args.get('num_folds', 10))
        self.random_flag = int(args.get('random_flag', 0))
        self.output_dir = args.get('output_dir', os.getcwd()).strip()
        self.num_lower_repeats = int(args.get('num_lower_repeats', 1))
        self.num_lower_folds = int(args.get('num_lower_folds', 10))
        self.classifier_name = args.get('classifier_name', 'Linear_SVM').strip()
        self.data_type = args.get('data_type', 'connectomics')
        self.data_type_args = args.get('data_type_args', {})
        self.dont_log = bool("dont_log" in args)
        self.log = sys.stdout
        self.merge_flag = args.get('merge_flag', 'ClassifierSmallest').strip()

        # Get/create output dir
        self.output_dir = os.path.join(self.output_dir, '')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print('output directory created!')
        else:
            print('output directory exists!')

        # Constant Internal Variables:

        # Performance measures:
        # balanced_accuracy, accuracy, sensitivity, specificity, f1, mcc,
        # ppv, coefficients
        self.num_performance_metrics = 7

        # Set during data load:
        self.data_source = None
        self.features = None  # the dataset
        self.feature_names = None # the feature column names
        self.num_features = None
        self.groups = None

        # Set during setup_folds:
        self.folds = list()  # a list of scikit-learn stratified k folds instances

        # A list of (train_indices, test_indices) tuples with len(self.folds)
        # This identifies the subjects to use as train and test samples for each
        # fold
        self.folds_structure = list()

        # Mutates each CV iteration
        self.train_partitions = None
        self.train_partitions_performance = None
        self.train_partition_index = None

        self.current_iteration = None
        self.test_train_performance = None
        self.test_train_partitions = None
        self.base_performance = None
        self.epic_performance = None

        self.active_features = None
        self.active_feature_names = None
        self.num_active_features = None

        # Used to find the mean coefficients for svms along each fold
        # Set mean = FALSE, to calc diff svm coef weights for each fold
        self.to_keep = None
        self.mean_weights = None
        self.use_mean = True

    def print_input_arguments(self, logfile=None):
        """
        Prints the arguments that define this run of epic tools.
        """
        if logfile is None:
            logfile = sys.stdout

        print('EPIC Parameters:', file=logfile)
        print('  info_table_name = ' + str(self.info_table_name), file=logfile)
        print('      num_repeats = ' + str(self.num_repeats), file=logfile)
        print('        num_folds = ' + str(self.num_folds), file=logfile)
        print('      random_flag = ' + str(self.random_flag), file=logfile)
        print('       output_dir = ' + self.output_dir, file=logfile)
        print('num_lower_repeats = ' + str(self.num_lower_repeats),
              file=logfile)
        print('  num_lower_folds = ' + str(self.num_lower_folds), file=logfile)
        print('  classifier_name = ' + self.classifier_name, file=logfile)
        print('       merge_flag = ' + self.merge_flag, file=logfile)
        print('\n', file=logfile, flush=True)

    def merge_optimize(self):
        """
        Run the epic classification algorithm using cortical thickness
        """

        if not self.dont_log:
            logname = os.path.join(self.output_dir,
                                   "EPIC_MERGE_OPTIMIZE_LOG.txt")
            self.log = open(logname, 'w')
            print("Logging to file ... log is located at {}".format(logname))
        try:
            # print out arguments
            self.print_input_arguments(logfile=self.log)

            # load data
            self.load_data()
            self.portion_control_data()
            # TODO: Group size portion control
            self.condition_data()
            #self.boruta_filter_data()
            self.allocate_epic_containers()

            # setup classifier
            self.setup_folds()

            # learn partition
            while self.has_cv_iterations():

                # Rank all features
                self.rank_features()

                # repeatedly merge regions
                while self.has_next_partition():
                    self.score_partition()

                # choose best merge amount and partition
                self.select_optimal_partition()

                # test and compare partition
                self.classify_test_data()

                self.update_cv_iterations()

            # save stats
            self.save_performances()
        finally:
            self.log.close()

    def rank_features(self):
        """Rank the features and create a partition spread"""

        # get training data
        train_features, train_groups = self.current_train_data()

        # compute weights
        clf = self.select_classifier()
        clf.fit(train_features, train_groups)
        coefs = copy.deepcopy(clf.coef_[0])

        if not self.use_mean:
            self.train_partitions = self.data_source.create_partitions(coefs)
        else:
            self.train_partitions = copy.copy(self.mean_partition)

        self.train_partitions_performance = np.zeros(
            (self.train_partitions.shape[0], self.num_performance_metrics)
        )

        self.train_partition_index = 0

    def update_train_partition_performance(self, partition_performance):

        self.train_partitions_performance[self.train_partition_index,
        :] = partition_performance

    def has_next_partition(self):

        return self.train_partition_index < self.train_partitions.shape[0]

    def current_train_partition(self):

        return self.train_partitions[self.train_partition_index, :]

    def load_data(self):
        """
        Initializes self.data_source and grabs features and groups by calling
        the data_source load() method.
        """
        if not self.data_type in EpicDataSource.registry:
            raise ValueError("There is no data source interface class '{}' "
                             "defined in epic.data. Cannot load data!"
                             .format(self.data_type))
        data_cls = EpicDataSource.registry[self.data_type]
        self.data_source = data_cls(self, self.data_type_args)

        res = self.data_source.load(self.info_table_name)
        self.features, self.feature_names, self.groups = res
        self.num_features = self.features.shape[1]
        assert len(self.feature_names) == self.num_features

        print('Data size:\n\t{}\tparticipants\n\t{}\tfeatures per participant'
              .format(self.features.shape[0], self.features.shape[1]),
              file=self.log, flush=True)

    def boruta_filter_data(self):
        """
        Runs Boruta on self.features and removes the features determined to not
        be significant.
        """
        print("Starting Boruta analysis with a RandomForest of depth 5 ...",
              file=self.log, flush=True)
        with Capturing() as output:
            # define random forest classifier -- utilising all cores and
            # sampling in proportion to y labels
            rf = RandomForestClassifier(n_jobs=-1,
                                        class_weight='auto',
                                        max_depth=5)

            # define Boruta feature selection method
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2)

            # find all relevant features
            feat_selector.fit(self.features, self.groups)

            new_feature_names = self.feature_names[
                feat_selector.support_ + feat_selector.support_weak_
            ]
            # call transform() on X to filter it down to selected features
            self.features = feat_selector.transform(self.features, weak=True)
            self.num_features = self.features.shape[1]
            assert(self.feature_names.shape[0] == self.num_features)

        print(output, file=self.log, flush=True)

    def portion_control_data(self):
        """
        loads your data and ensures there is a 'good enough' proportion
        between the groups. If control group size is > 3 * x disease, then
        the data is modified to ensure the ration of disease vs control is 3:1

        Also calculates max_num_folds to ensure that the test groups for each
        fold have at a minimum approximately 3 disease members.
        """
        cn_v_dx_max_ratio = 3
        min_dx_per_test = 3

        grps, len_grps = np.unique(self.groups, return_counts=True)
        assert len(grps) == 2 and set(grps) == {0, 1}, \
               "Can only have two subject groups, control==0, and diseased==1"

        # if len(control_grp) > max_proportion * min_len:
        #     random_shuffle_control_grp_indices()
        #     control_grp = control_grp[:max_proportion * min_len]
        #
        # self.data = control_grp + disease_grp
        #
        # fold_limit = float(len(self.data)) / min_fold_size
        #
        # if num_folds < 1.0:
        #     raise ValueError("There are not enough samples to perform EPIC!")
        # elif int(num_folds) > 10:
        #     self.num_folds = 10

    def condition_data(self):
        """transform and scale features"""

        robusto = preprocessing.RobustScaler()
        self.features = robusto.fit_transform(self.features)
        # scale feature data
        #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        #self.features = min_max_scaler.fit_transform(self.features)

    def has_cv_iterations(self):

        return self.current_iteration[0] < self.num_repeats and \
               self.current_iteration[1] < self.num_folds

    def current_test_data(self, from_active = True):

        if from_active:
            features = self.active_features
        else:
            features = self.features
        test_features = features[
                        self.folds_structure[self.current_iteration[0]][
                            self.current_iteration[1]][1], :]
        test_groups = self.groups[
            self.folds_structure[self.current_iteration[0]][
                self.current_iteration[1]][1]]

        return test_features, test_groups

    def current_train_data(self):

        features = self.active_features

        assert self.to_keep is not None

        train_features = features[
                         self.folds_structure[self.current_iteration[0]][
                             self.current_iteration[1]][0], :]
        train_groups = self.groups[
            self.folds_structure[self.current_iteration[0]][
                self.current_iteration[1]][0]]

        return train_features, train_groups

    def update_cv_iterations(self):

        if self.current_iteration[1] < self.num_folds - 1:
            self.current_iteration[1] += 1
        else:
            self.current_iteration[1] = 0
            self.current_iteration[0] += 1

        print("Starting Iteration ({}, {})..."
              .format(self.current_iteration[1],
                      self.current_iteration[0]), file=self.log, flush=True)

    def allocate_epic_containers(self):

        self.current_iteration = [0, 0]

        # performance on training data
        self.test_train_performance = np.zeros(
            (self.num_repeats, self.num_folds, self.num_performance_metrics))
        self.test_train_partitions = {
            repeat_num: {
                fold_num: None for fold_num in range(self.num_folds)
                }
            for repeat_num in range(self.num_repeats)
            }

        self.base_performance = np.zeros(
            (self.num_repeats, self.num_performance_metrics))
        self.epic_performance = np.zeros(
            (self.num_repeats, self.num_performance_metrics))
        self.base_coefs = [{} for idx in range(self.num_repeats)]
        self.epic_coefs = [{} for idx in range(self.num_repeats)]

    def update_test_partition_test_stats(self,
                                         base_performance,
                                         base_coef_dict,
                                         epic_performance,
                                         epic_coef_dict):

        self.base_performance[self.current_iteration[0], :] = base_performance
        self.base_coefs[self.current_iteration[0]] = base_coef_dict
        self.epic_performance[self.current_iteration[0], :] = epic_performance
        self.epic_coefs[self.current_iteration[0]] = epic_coef_dict

    def update_test_partition_train_stats(self, performance, partition):

        self.test_train_performance[self.current_iteration[0]][
                                    self.current_iteration[1]] = performance
        self.test_train_partitions[self.current_iteration[0]][
                                   self.current_iteration[1]] = partition

    def get_current_train_optimal_partition(self):

        return self.test_train_partitions[self.current_iteration[0]][
                                          self.current_iteration[1]]

    def select_classifier(self):

        if self.classifier_name == 'Linear SVM' or self.classifier_name == 'Linear_SVM':
            clf = svm.LinearSVC(class_weight='balanced', C=10, dual=False,
                                penalty='l1', random_state=1, loss='squared_hinge')
        elif self.classifier_name == 'RBF SVM' or self.classifier_name == 'RBF_SVM':
            clf = svm.SVC(kernel='rbf', class_weight='balanced')
        elif self.classifier_name == 'LDA' or self.classifier_name == 'lda':
            clf = LinearDiscriminantAnalysis()
        elif self.classifier_name == 'LR':
            clf = LogisticRegression()

        return clf

    def score_partition(self):
        """
        Takes the current train features and train groups, creates "partitioned"
        feature data, where a "partitioned" feature data is the train features
        modified such that features that are merged within the current
        train_partition_index are averaged within their merged column and set to
        zero within their other column(s).

        Then evaluates classifiers self.num_lower_repeats *
        self.num_lower_folds times, and saves the average performance for
        those classifiers within self.train_partitions_performance for the
        current index.
        """
        train_features, train_groups = self.current_train_data()

        p = self.partition_matrix(self.current_train_partition())

        partitioned_features = np.zeros((train_features.shape[0], p.shape[1]))

        for i in range(0, train_features.shape[0]):
            partitioned_features[i, :] = train_features[i, :].dot(p)

        partition_performance = self.evaluate_classifier(partitioned_features,
                                                         train_groups)
        self.update_train_partition_performance(partition_performance)

        self.train_partition_index += 1

    def select_optimal_partition(self):

        optimal_index = np.argmax(self.train_partitions_performance[:, 1])
        self.update_test_partition_train_stats(
            self.train_partitions_performance[optimal_index, :],
            self.train_partitions[optimal_index, :])

    def classify_test_data(self):

        train_features, train_groups = self.current_train_data()
        test_features, test_groups = self.current_test_data()
        names = self.active_feature_names
        # baseline performance
        base_performance, base_coef_dict = self.evaluate_single_classifier(
            names,
            train_features,
            train_groups,
            test_features,
            test_groups
        )

        opt_partition = self.get_current_train_optimal_partition()
        p = self.partition_matrix(opt_partition)

        partitioned_train_features = np.zeros(
            (train_features.shape[0], p.shape[1]))
        partitioned_test_features = np.zeros(
            (test_features.shape[0], p.shape[1]))

        for i in range(0, train_features.shape[0]):
            partitioned_train_features[i, :] = train_features[i, :].dot(p)

        for i in range(0, test_features.shape[0]):
            partitioned_test_features[i, :] = test_features[i, :].dot(p)

        names_idx, lookup = np.unique(opt_partition.astype(int),
                                      return_inverse=True)
        assert lookup.shape[0] == names.shape[0]
        merge_names = [[] for idx in range(names_idx.shape[0])]
        for i in range(lookup.shape[0]):
            merge_names[lookup[i]].append(names[i])

        for i in range(len(merge_names)):
            merge_names[i] = "+".join(merge_names[i])

        # epic performance
        epic_performance, epic_coef_dict = self.evaluate_single_classifier(
            merge_names,
            partitioned_train_features,
            train_groups,
            partitioned_test_features,
            test_groups
        )

        # update test data stats
        self.update_test_partition_test_stats(base_performance,
                                              base_coef_dict,
                                              epic_performance,
                                              epic_coef_dict)

    def evaluate_single_classifier(self,
                                   feature_names,
                                   train_features,
                                   train_groups,
                                   test_features,
                                   test_groups):

        clf = self.select_classifier()

        ac_list = np.zeros(self.num_performance_metrics)

        # train svm
        clf.fit(train_features, train_groups)

        # evaluate test data
        predictions = clf.predict(test_features)

        # compute performance measures
        ac_list = self.performance_measures(test_groups, predictions)
        assert(clf.coef_.shape[1] == len(feature_names))
        coef_dict = dict(zip(feature_names, clf.coef_[0, :]))
        return ac_list, coef_dict

    rfe_threshold = 0.08
    rfe_threshold_2 = 0.05

    def setup_folds(self):
        """
        Breaks up data into StratifiedKFolds num_repeats times.
        """
        for i in range(0, self.num_repeats):
            skf = StratifiedKFold(self.groups,
                                  int(self.num_folds),
                                  shuffle=True,
                                  random_state=i + self.random_flag)
            self.folds.append(skf)

            repeat_list = list()

            for train_index, test_index in skf:
                import pdb; pdb.set_trace()
                repeat_list.append((train_index, test_index))

            self.folds_structure.append(repeat_list)

        classifier = self.select_classifier()
        # Recursive Feature Elimination, flat-style. Runs classifier
        # twice to eliminate silly features.
        coef1_struct = []
        # Initial svms
        for i in range(0, self.num_repeats):
            train_idxs = self.folds_structure[i][0][0]
            train_features = self.features[train_idxs, :]
            train_groups = self.groups[train_idxs]
            classifier.fit(train_features, train_groups)
            coef1_struct.append(copy.deepcopy(classifier.coef_[0]))

        coef1_array = np.array(coef1_struct)
        coef1_mean = np.mean(coef1_array, axis=0)

        _to_keep = []
        _to_toss = []
        for idx, wgt in enumerate(coef1_mean):
            if abs(wgt) > self.rfe_threshold:
                _to_keep.append(idx)
            else:
                _to_toss.append(idx)

        coef2_struct = []
        for i in range(0, self.num_repeats):
            train_idxs = self.folds_structure[i][0][0]
            train_features = self.features[train_idxs, :]
            train_features = train_features[:,_to_keep]
            train_groups = self.groups[train_idxs]
            classifier.fit(train_features, train_groups)
            coef2_struct.append(copy.deepcopy(classifier.coef_[0]))

        coef2_array = np.array(coef2_struct)
        coef2_mean = np.mean(coef2_array, axis=0)

        _to_keep = []
        for idx, wgt in enumerate(coef2_mean):
            if abs(wgt) > self.rfe_threshold_2:
                _to_keep.append(idx)

        to_keep = []
        mean_weights = []
        _to_toss.reverse()
        incr = 0
        next_toss = _to_toss and _to_toss.pop() or -1
        # Re-create the indices for the full feature set based on the
        # double feature elimination routine above
        for idx in _to_keep:
            keep_idx = idx + incr
            while _to_toss and keep_idx >= next_toss:
                incr += 1
                next_toss = _to_toss.pop()
                keep_idx = idx + incr
            to_keep.append(keep_idx)
            mean_weights.append(coef2_mean[idx])

        self.to_keep = to_keep
        self.mean_weights = mean_weights
        self.active_features = self.features[:, self.to_keep]
        self.active_feature_names = self.feature_names[self.to_keep]
        self.num_active_features = len(self.active_feature_names)
        self.mean_partition = self.data_source.create_partitions(self.mean_weights)

    def partition_matrix(self, partition):

        # number of regions in the partition
        labels, labels_index = np.unique(partition, return_inverse=True)
        p = np.zeros((len(partition), len(labels)))
        p[np.arange(0, len(partition)), labels_index] = 1

        for i in range(0, len(labels)):
            p[:, i] = p[:, i] / np.sum(p[:, i])

        return p

    def evaluate_classifier(self, features, groups):
        """Evalute the selected classifier"""

        ac_list = np.zeros(
            (self.num_lower_repeats, self.num_performance_metrics))
        clf = self.select_classifier()

        for i in range(0, self.num_lower_repeats):
            skf = StratifiedKFold(groups, int(self.num_lower_folds),
                                  shuffle=True)

            single_ac_list = np.zeros(
                (self.num_lower_folds, self.num_performance_metrics))
            single_index = 0

            for train_index, test_index in skf:
                # train svm
                clf.fit(features[train_index, :], groups[train_index])

                # evaluate test data
                predictions = clf.predict(features[test_index, :])

                single_ac_list[single_index, :] = self.performance_measures(
                    groups[test_index], predictions)
                single_index = single_index + 1

            ac_list[i, :] = np.mean(single_ac_list, 0)

        return np.mean(ac_list, 0)

    def performance_measures(self, y_true, y_pred):

        tp = np.count_nonzero((y_pred == 1) & (y_true == 1))
        tn = np.count_nonzero((y_pred == 0) & (y_true == 0))
        fp = np.count_nonzero((y_pred == 1) & (y_true == 0))
        fn = np.count_nonzero((y_pred == 0) & (y_true == 1))

        if (tp + fn) > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0

        if (tn + fp) > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0

        accuracy = (tp + tn) / (tp + fn + tn + fp)
        balanced_accuracy = (sensitivity + specificity) / 2

        f1 = (2 * tp) / (2 * tp + fp + fn)
        if (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) == 0:
            mcc = 0
        else:
            mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        ppv = bool(tp + fp) and (tp)/(tp + fp) or 0

        performance = np.zeros((1, self.num_performance_metrics))
        performance[0, 0] = balanced_accuracy
        performance[0, 1] = accuracy
        performance[0, 2] = sensitivity
        performance[0, 3] = specificity
        performance[0, 4] = f1
        performance[0, 5] = mcc
        performance[0, 6] = ppv

        return performance

    def save_performances(self):

        self.save_performance(self.base_performance,
                              self.base_coefs,
                              '\n\n'
                              'Base Performance Results:\n'
                              '-------------------------\n')
        self.save_performance(self.epic_performance,
                              self.epic_coefs,
                              '\n\n'
                              'EPIC Performance Results:\n'
                              '-------------------------\n')

    def save_performance(self, ac_list, coef_list, results_title):

        print(results_title, file=self.log)
        print('\n' + str(self.num_repeats) + ' times repeated ' + str(
            self.num_folds) + ' fold cross validation using ',
              self.classifier_name, ':\n', file=self.log)
        print('Repeat Results:', file=self.log)
        print(',\t'.join(['#', "Balanced Accuracy", "Accuracy",
                          "Sensitivity", "Specificity", "F1", "MCC", "PPV"]),
              file=self.log)

        for i in range(0, self.num_repeats):
            row_vals = [str(i)] + [str(metric) for metric in ac_list[i,:]]
            print("{}".format(",\t".join(row_vals)), file=self.log)
            print("\nFeature names and classifier coefficients", file=self.log)
            print("\t{}".format(coef_list[i]), file=self.log)

        print('\n\nMean Results:', file=self.log)
        print('Balanced Accuracy, Accuracy, Sensitivity, Specificity, F1, MCC, PPV',
              file=self.log)

        for j in range(0, self.num_performance_metrics - 1):
            print(str(np.mean(ac_list, 0)[j]) + ', ', end='', file=self.log)
        print(str(np.mean(ac_list, 0)[self.num_performance_metrics - 1]),
              file=self.log)

        print('\n\nStandard Deviation of Results:', file=self.log)
        print('Balanced Accuracy, Accuracy, Sensitivity, Specificity, F1, MCC, PPV',
              file=self.log)

        for j in range(0, self.num_performance_metrics - 1):
            print(str(np.std(ac_list, 0)[j]) + ', ', end='', file=self.log)
        print(str(np.std(ac_list, 0)[self.num_performance_metrics - 1]),
              file=self.log)
