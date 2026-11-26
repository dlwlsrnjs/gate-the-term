# Camera-ready integration and claim limits

This package supplies new controlled experiments. Provisional completed-run outcomes are now available in [results/latest/RESULTS_KO.md](results/latest/RESULTS_KO.md), with per-item numeric records and source checksums. All six primary arms have completed seed 1; scalar GRPO, separated normalization, strict CGH, and pivot RSFT have also completed seed 2 (ten runs total). The [strict/pivot analysis](results/CGH_PIVOT_KO.md) documents the direct comparisons and English-access limitations. Full method-level seed means remain withheld until all three planned seeds are present. Source-host smoke/timing outputs remain functionality checks, not additional accuracy results.

In seed 1, strict obtains 68/180 versus 55/180 for GRPO and split, 59/180 for pivot RSFT, and 72/180 for unweighted selective updates. Strict’s +5.00 pp versus pivot has a conditional problem-cluster interval of [−2.22, +12.22] pp, excluding training-seed uncertainty and multiplicity adjustment. The strict–pivot gain is concentrated in Thai; pivot is better in Bengali and Telugu. These observations do not establish strict’s superiority over reference distillation, the necessity of weighting, or the pure causal effect of English content.

Pivot RSFT seed 2 obtains 60/180 (Bengali/Telugu/Thai: 21/16/23), versus 59/180 (21/12/26) in seed 1. Its matched-GRPO gain is +2.22 pp and +2.78 pp, respectively; the Bengali/Telugu improvements and Thai decrease occur in both seeds. The seed-2 fixed-model problem-cluster interval versus GRPO is [−4.44, +10.00] pp. Strict seed 2 obtains 68/180 (23/3/42), matching its seed-1 total. Its +7.22 pp over matched GRPO and split repeats in both seeds; seed-2 conditional intervals are [+2.22, +12.22] and [+2.78, +11.67] pp, respectively. Strict–pivot is +4.44 pp in seed 2 with interval [−3.33, +12.22] pp. The Telugu/Thai trade-off repeats, while the Bengali direction flips. Two completed seeds do not establish general robustness or strict superiority over distillation. These observations neither identify a pure English-content effect nor support comparing strict seed 1 against pivot seed 2. See the [seed comparison](results/latest/SEED_CONSISTENCY_KO.md).

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

After the prespecified cohort completes, include a compact primary-control table in the main paper and the complete per-seed/per-language results in the appendix. Include the external benchmark table with its provenance, overlap exclusions, and clustered intervals. Do not insert pending dashes into a final camera-ready result table.
