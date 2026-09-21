# Gate failure handling

The first micro experiment (results/compile_boundary_20260918_233238) failed the predeclared exact-equality gate for all nine cases. Native4 kernels versus joint2 is confirmed, with finite but unequal outputs. Keep those raw results unchanged. The experiment source files used for that run are unchanged; a new model_study.py driver is added for this diagnostic extension, with a separate manifest/result directory.

Continue the planned same-history numerical diagnostics and stage timelines for all9 cases, in separate native/joint processes. Native sampler work remains in timeline captures; returned IDs are forced on the host to the saved native history. Hash full logits at every step, and compare complete logits plus newly written cache slots across all28 layers at prefill, early decode, final decode, and page-boundary steps. This is not a model quality evaluation.

Do not run the planned10 performance pairs because correctness gate failed. Do not relax tolerances or claim an accepted model replacement. No handwritten kernel is implemented. The hotspot evidence remains useful to choose a kernel project; lower local duration remains diagnostic only.

Source inspection shows Inductor removed intermediate BF16 rounding across the new Norm/RoPE boundary. This is a concrete changed computation; it does not prove that rounding is the only contributor to every output difference.
