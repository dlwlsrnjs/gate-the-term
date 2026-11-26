# Camera-ready integration and claim limits

The complete local primary cohort comprises six arms × three seeds (18 runs), all finished with final MGSM evaluation by 2026-11-26 18:54 KST. See [three-seed results](results/latest/THREE_SEED_RESULTS_KO.md), [per-seed outcomes and provenance](results/latest/RESULTS_KO.md), and [reviewer interpretation](results/REVIEWER_INTERPRETATION_KO.md). Secondary training (nine runs) and external evaluation (19 models) remain uncollected here; local completion does not imply their completion.

Mean language-gated accuracy is 30.56% for GRPO, 28.89% for split, 37.41% for strict CGH, 31.85% for pivot RSFT, 38.33% for legacy, and 38.89% for unweighted selective updates. Strict exceeds GRPO and split in all three seeds. Exploratory crossed seed/problem intervals for strict−GRPO and strict−split are [+2.59, +11.11] and [+4.07, +13.15] pp. Strict−pivot is +5.56 pp with interval [−1.11, +12.41] pp, so superiority over reference distillation remains uncertain. Only three seeds support the empirical seed resampling; intervals are not multiplicity-adjusted.

Pivot shows Bengali/Telugu gains and Thai losses versus GRPO in every seed. Strict is below pivot in Telugu and above it in Thai in every seed; its Bengali difference flips. NoIS exceeds strict in all three seeds, which does not establish a need for weighting. Split matched GRPO in the first two seeds but lost 5.00 pp in the third; report the full cohort rather than freezing the early equal-score interpretation.

## Main claims

Frame the contribution as a mismatch between scalar and component-level diagnostics, plus a practical operator-dependent intervention. A constant component has zero contribution to a realized centered advantage. This does not mean that GRPO destroys an existing population gradient or cannot learn from more samples.

An appropriate title direction is “Gate the Term, Not the Group: Component-Level Signal Starvation in Group-Relative RL.” The existing working title emphasizing diagnosis is also consistent with the evidence. Avoid promising exact gradient restoration in the deployed implementation.

The exact sequence-ratio policy-gradient identity is standard and applies to an ideal, unfiltered term. Clipping, root ratios, mean-token scores, gating, and admission change the implemented estimator. The actual surrogate is biased. A one-block finite-sample cosine, especially with low ESS, cannot certify a full-model or population gradient.

## Questions answered by the six primary controls

| Comparison | Interpretation |
|---|---|
| `vanilla` vs `split` | Effect of separated normalization |
| `split` vs `cgh_strict` | Selective target replacement with fixed original auxiliary coefficients |
| `cgh_legacy` vs `cgh_strict` | Historical coefficient branches versus strict construction in the same engine; several branches differ, so this is not an auxiliary-only ablation |
| `pivot_rsft` vs `cgh_strict` | Stronger information-access control using correct pivot-assisted responses under the original question |
| `cgh_strict_nois` vs `cgh_strict` | Contribution of the practical clipped/root weighting |
| `pivot_rsft` vs `cgh_strict_nois` | Always-positive versus all-wrong-selective extra admission with weight one and the same split original-response update; update counts and resulting trajectories differ |
| `vanilla` vs `cgh_legacy` | Clean-engine analogue of the historical headline comparison |

These are fresh exploratory experiments. All arms use the same engine and response count, but actual token/FLOP and backward costs differ. Primary arms generate reference-assisted extras under the same conditioning; vanilla and split discard them. They do not replay an identical frozen response pool. A pure English-content effect would require a q-only proposal with the same objective/admission; q_resample changes both proposal and update. Do not decompose the total strict gain into additive English and estimator effects. An advantage over a weak relative-pivot control alone does not establish that a gate is necessary. If pivot RSFT performs similarly, report that the diagnosis remains useful while the necessity of the specific estimator is unproven.

Secondary comparisons test whole-estimator gating, original-question resampling, and component-triggered entropy shaping. Label adaptations accurately; they are not full DAPO/RL-ZVP reproductions.

## Evaluation and uncertainty

Keep historical and fresh cohorts separate. The historical three-seed evaluation reused 50 MGSM source IDs in three languages, not 450 independent questions. The historical six-seed table reused those same IDs. A language's gain averaged over seeds is not that many new independent test successes.

The fresh MGSM holdout has 60 IDs: only ten more than before. The substantial expansion is external: 1,069 new Indic-GSM8K IDs and 1,000 MSVAMP IDs. Report per-dataset and per-language values, raw numerical correctness, language-gated correctness, and script consistency. The MSVAMP release has no Telugu subset.

Cluster translations by source ID and resample seeds and IDs separately. Present confidence intervals with the selected languages fixed. Three seeds still provide limited evidence about training-run variability, even with more test items. Do not select checkpoints, seeds, or metrics after examining favorable results.

## Limits to retain

- English parallel reference access is privileged training information. More evaluation questions do not remove that confound; the pivot RSFT control addresses it.
- Results remain arithmetic evaluations on one base model. Evaluating transfer to MSVAMP does not establish trained repair in safety or unrelated domains.
- Eight failed proposal draws do not establish zero success probability at every larger budget.
- Clipped weights are bounded by construction. Inspect pre-clipping weights and exact-ratio ESS separately.
- Lower baseline accuracy and larger observed gains in three languages do not establish a general causal relationship.

The local primary cohort is complete: include its six-arm table in the main paper and the complete per-seed/per-language results in the appendix. Include the external benchmark table with its provenance, overlap exclusions, and clustered intervals. Do not insert pending dashes into a final camera-ready result table.
