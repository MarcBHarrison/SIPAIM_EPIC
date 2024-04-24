NUM_PERFORMANCE_METRICS = 7
"""
Performance measures are:
1. balanced_accuracy
2. accuracy
3. sensitivity
4. specificity
5. f1
6. mcc
7. ppv
"""

MAX_COMBINATORIAL = 16
"""Max number of features to take of a feature type when calculating AIC for
partition type combinatorial sets

"""

MAX_MERGES = 3
"""Max number of partition merge sets to take for a partition set. Total number
of partition rows is equal to MAX_MERGES^12, as there are 6 partition sets --
dx_vl, dx_thk, dx_sa, sex_vl, sex_thk, sex_sa. DON'T RAISE PAST 3 unless you have
max combinatorial set to < 12 AND no sex partition sets. 
3 will take a long time! Recommended is 2!

"""

MIN_DIFF_THRESH = 0.5
"""TBD. This threshold is the minimum value taken for normalized 
difference equation results. The formulas are basically t-tests, so values > 2 are 
closer to approaching significant. Often the max combinatorial will choose fewer features than the number
that meet the min_diff_thresh, so it's usually not necessary to change this too much.
"""

PREDICTION_HOLDOUT_RATIO=0.25
"""Reserve 25% of the samples to validate final models against. This should change
depending on your sample size."""
