=======
History
=======

0.1.6 (2023-01-23)
-------------------------------

* `Aggregators`:
  * Possibility to add or not the current ratings to the training set.

* `Embeddings`:

  * The parameter `norm` has no default value (instead of `True`).
  * Fix a bug: when `norm=False`, the values of the attributes `n_voter` and `n_dim` were swapped by mistake.
  * Rename method `scored` to `times_ratings_candidate`.
  * Rename method `_get_center` to `get_center`, so that it is now part of the API.
  * Rename method `normalize` to `normalized`, `recenter` to `recentered`, `dilate` to `dilated` because they
    return a new `Embeddings` object (not modify the object in place).
  * Fix a bug in method `get_center`.
  * Methods `get_center`, `recentered` and `dilated` now also work with non-normalized embeddings.
  * Document that `dilated` can output embeddings that are not in the positive orthant.
  * Add `dilated_new`: new dilatation method whose output is in the positive orthant.
  * Add `recentered_and_dilated`: recenter and dilate the embeddings (using `dilated_new`).
  * Add `mixed_with`: mix the given `Embeddings` object with another one.
  * Rename `plot_scores` to `plot_ratings_candidate`.

* Embeddings generators:

  * Rename `EmbeddingsGeneratorRandom` to `EmbeddingsGeneratorUniform`.
  * Add `EmbeddingsGeneratorFullyPolarized`: create embeddings that are random vectors of the canonical basis.
  * `EmbeddingsGeneratorPolarized` now relies on `EmbeddingsGeneratorUniform`, `EmbeddingsGeneratorFullyPolarized`
    and the method `Embeddings.mixed_with`.
  * Move `EmbeddingCorrelation` and renamed it.
  * Rewrote the `EmbeddingsFromRatingsCorrelation` and how it compute the number of singular values to take.

* Epistemic ratings generators:

  * Add `TruthGenerator`: a generator for the ground truth ("true value") of each candidate.
  * Add `TruthGeneratorUniform`: a uniform generator for the ground truth ("true value") of each candidate.
  * `RatingsGeneratorEpistemic` and its subclasses now take a `TruthGenerator` as parameter.
  * Add `RatingsGeneratorEpistemicGroups` as an intermediate class between the parent class `RatingsGeneratorEpistemic`
    and the child classes using groups of voters.
  * `RatingsGeneratorEpistemic` now do not take groups sizes as parameter: only `RatingsGeneratorEpistemicGroups`
    and its subclasses do.
  * Rename `RatingsGeneratorEpistemicGroupedMean` to `RatingsGeneratorEpistemicGroupsMean`,
    `RatingsGeneratorEpistemicGroupedMix` to `RatingsGeneratorEpistemicGroupsMix`
    `RatingsGeneratorEpistemicGroupedNoise` to `RatingsGeneratorEpistemicGroupsNoise`.
  * Remove method `RatingsGeneratorEpistemic.generate_true_values`: the same result can be obtained with
    `RatingsGeneratorEpistemic.truth_generator`.
  * Add `RatingsGeneratorEpistemicGroupedMixFree` and `RatingsGeneratorEpistemicGroupsMixScale`.

* Ratings generators:

  * `RatingsGenerator` and subclasses: remove `*args` in call because it was not used.
  * `RatingsGeneratorUniform`: add optional parameters `minimum_rating` and `maximum_rating`.
  * Possibility to save scores in a csv file

* `RatingsFromEmbeddingsCorrelated`:

  * Move parameter `coherence` from `__call__` to `__init__`.
  * Rename parameter `scores_matrix` to `ratings_dim_candidate`.
  * Parameters `n_dim` and `n_candidates` are optional if `ratings_dim_candidate` is specified.
  * Add optional parameters `minimum_random_rating`, `maximum_random_rating` and `clip`.
  * Parameter `clip` now defaults to `False` (the former version behaved as if `clip` was always True).

* Single-winner rules:

  * Rename `ScoringRule` to `Rule`.
  * Rename all subclasses accordingly. For example, rename `FastNash` to `RuleFastNash`.
  * Rename `SumScores` to `RuleSumRatings` and `ProductScores` to `RuleProductRatings`.
  * Rename `RulePositionalExtension` to `RulePositional` and rename subclasses accordingly.
  * Rename `RuleInstantRunoffExtension` to `RuleInstantRunoff`.
  * Add `RuleApprovalSum`, `RuleApprovalProduct`, `RuleApprovalRandom`.
  * Changed the default renormalization function in `RuleFast`.
  * Change the method in `RuleMLEGaussian`.
  * Add `RuleModelAware`.
  * Add `RuleRatingsHistory`.
  * Add `RuleShiftProduct` which replace `RuleProductRatings`.

* Multiwinner rules: rename all rules with prefix `MultiwinnerRule`. For example, rename `IterFeatures` to
  `MultiwinnerRuleIterFeatures`.

* Manipulation:

  * Rename `SingleVoterManipulation` to `Manipulation` and rename subclasses accordingly.
  * Rename `SingleVoterManipulationExtension` to `ManipulationOrdinal` and rename subclasses accordingly.
  * Rename `ManipulationCoalitionExtension` to `ManipulationCoalitionOrdinal` and rename subclasses accordingly.

* Rename `AggregatorSum` to `AggregatorSumRatings` and `AggregatorProduct` to `AggregatorProductRatings`.
* Add `max_angular_dilatation_factor`: maximum angular dilatation factor to stay in the positive orthant.
* Rename `create_3D_plot` to `create_3d_plot`.
* Moved function to the utils module.
* Reorganize the file structure of the project.

0.1.5 (2022-01-04)
------------------

* Aggregator functions.
* Online learning.
* Refactoring Truth epistemic generators.
* Rule taking history into account.

0.1.4 (2021-12-06)
------------------

* New version with new structure for Ratings and Embeddings

0.1.3 (2021-10-27)
------------------

* New version with new internal structure for the library

0.1.2 (2021-07-05)
------------------

* New version with handy way to use the library for algorithm aggregation and epistemic social choice


0.1.1 (2021-04-02)
------------------

* Minor bugs.

0.1.0 (2021-03-31)
------------------

* End of the internship, first release on PyPI.

