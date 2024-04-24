*******
EPIC v2
*******

This is an overview of a proposed new EPIC algorithm:


1. loads your data and ensures there is a 'good enough' proportion
   between the groups. If control group size is > 3x disease, then
   the data is modified such that::

      max_proportion = 3 # max ratio of one group compared to the other group
      min_disease_samples_per_test = 3

      min_len = len(disease_grp)

      if len(control_grp) > max_proportion * min_len:
          random_shuffle_control_grp_indices()
          control_grp = control_grp[:max_proportion * min_len]

      self.data = control_grp + disease_grp

      fold_limit = float(len(self.data)) / min_fold_size

      if num_folds < 1.0:
          raise ValueError("There are not enough samples to perform EPIC!")
      elif int(num_folds) > 10:
          self.num_folds = 10

2. Run Boruta to narrow down the variables. Remove all the
   "definitely not important" variables.

3. Run the classifier once.

   1. Store its results as prev_best_classifier.
   2. Create a variable best_classifier_found and set it to 0

4. Iteratively prune the variables as follows:

   1. Rank the variables according to their contribution to the
      classifier.
   2. Make two new variable sets:

      1. Take the lowest ranked variable and merge it with the
         next lowest within imaging class.
      2. Take the lowest ranked variable and remove it.

   3. If the new variable sets have variable len <
      min_variable_len, finish the analysis using the current best
      classification.

   4. Run both new variable sets through the classifier.

   5. Select the best classifier based off of which of the three
      available classifiers provides the better overall
      classification.

      1. If the previous best classifier is still best, increment
         best_classifier_found by 1, and pick the next best
         classifier to run in the next iteration.

      2. If best_classifier_found is >= 2, then finish the
         analysis.

      3. Else, set prev_best classifier to the new best, use the
         new best classifier as the classifier for the next
         iteration, and reset best_classifier_found to 0.

