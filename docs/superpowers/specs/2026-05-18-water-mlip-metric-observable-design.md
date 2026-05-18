# Do Force Errors Predict Liquid-Water MD?

## Summary

This project will test whether standard machine-learned interatomic potential (MLIP) validation metrics predict physically meaningful molecular dynamics (MD) behavior for liquid water. The project will use the public Cheng et al. revPBE0-D3 water training data and train several checkpoints of one MLIP architecture on the same dataset. Each checkpoint will be evaluated both by static energy/force errors and by downstream liquid-water MD observables.

The intended output is not a new best-in-class water potential. The intended output is a reproducible benchmark that maps static MLIP metrics to liquid-water MD quality.

## Research Question

When a water MLIP achieves lower energy and force error on held-out configurations, does that improvement reliably translate into better liquid-water MD observables?

The project will specifically look for cases where static validation metrics improve smoothly while physical observables improve non-monotonically, saturate, or fail.

## Motivation

MLIPs are usually trained and selected with energy and force errors on validation or test configurations. For MD, those pointwise errors may be insufficient because trajectories accumulate errors over time and physical observables depend on the shape of the sampled free-energy landscape. Water is a strong first system because it is small, heavily studied, and sensitive to structural, dynamical, and thermodynamic errors.

The project is motivated by the following literature themes:

- "Forces are not Enough" argues that force-field usefulness must be evaluated through molecular simulations, not only force accuracy.
- "How to validate machine-learned interatomic potentials" argues for physically guided validation beyond RMSE and MAE.
- Recent water MLIP studies show that modern equivariant models can produce stable and transferable water potentials, but also highlight the importance of validation against liquid and phase observables.
- Data-generation reviews emphasize that training-set coverage, not just model architecture, controls MLIP reliability.

## Data Source

The project will use the Cheng et al. revPBE0-D3 water dataset as the anchor dataset.

Primary sources:

- Materials Cloud record: `materialscloud:2018.0020/v1`
- GitHub resource: `BingqingCheng/neural-network-potential-for-water-revPBE0-D3`
- Related paper: Cheng, Engel, Behler, Dellago, and Ceriotti, "Ab initio thermodynamics of liquid and solid water"

This dataset is preferred because it is public, compact, cited, includes energies and forces, and is directly tied to previous water MLIP work. The first project version will avoid generating new DFT data.

## Default Modeling Choice

The default first architecture will be MACE, assuming no practical blocker appears during implementation planning.

Reasons:

- It is a modern equivariant MLIP architecture.
- It has accessible training tooling.
- It integrates naturally with ASE workflows for MD.
- It is a practical first choice for a reproducible side project.

NequIP or Allegro remain reasonable alternatives, but they are not the default for the first version because architecture comparison is outside the initial scope.

## Experimental Design

The experiment will train one model family on one dataset and evaluate multiple saved checkpoints. The checkpoints should span a range of model quality, such as:

- early underfit checkpoint
- intermediate checkpoint
- near-converged checkpoint
- best validation-force checkpoint
- optional late checkpoint if overfitting is visible

All checkpoints will use the same data split and evaluation protocol. This controls the comparison so the main variable is model quality rather than architecture, data source, or simulation workflow.

## Static Validation Metrics

Each checkpoint will be evaluated on held-out configurations with:

- energy MAE and RMSE
- force MAE and RMSE
- force error distribution tails
- optional per-species force errors for hydrogen and oxygen
- optional environment-stratified errors if practical

The benchmark should preserve more than average errors because rare large errors may be more predictive of MD failure than mean error.

## MD Observables

Each checkpoint will be deployed into an identical liquid-water MD protocol. The first protocol will use fixed-density liquid water near ambient conditions for structural and dynamical observables. A short NVE segment will test energy conservation after equilibration. If the checkpoint is stable, a secondary NPT check will estimate equilibrium density.

The initial observable set will include:

- short NVE energy drift
- trajectory stability and absence of unphysical bond distortions
- O-O radial distribution function
- O-H radial distribution function
- H-H radial distribution function
- diffusion coefficient from mean-squared displacement
- NPT density estimate for stable checkpoints

The first implementation should prioritize robust, interpretable observables over a long list of fragile analyses.

## Analysis

The analysis will compare static validation metrics against MD observables across checkpoints.

Primary comparisons:

- force MAE versus RDF error
- force MAE versus diffusion error
- force MAE versus NVE drift
- energy error versus liquid observables
- force error tail metrics versus instability or observable errors

The final result should identify which validation metrics, if any, are useful early predictors of liquid-water MD quality.

## Scope

In scope:

- using an existing public water dataset
- training one MLIP architecture
- evaluating multiple checkpoints from the same training run or controlled repeated runs
- running liquid-water MD with a fixed protocol
- computing static metrics and core water observables
- producing reproducible scripts and a written analysis

Out of scope for the first version:

- generating new DFT data
- comparing many architectures
- active learning
- nuclear quantum effects
- vapor-liquid equilibrium
- ice transferability
- dielectric constants
- electrolyte solutions
- publication-quality optimization of the best water potential

## Success Criteria

The project is successful if it produces:

- a reproducible dataset preparation path from the Cheng et al. data
- trained MLIP checkpoints with a measurable range of static validation quality
- MD trajectories for each checkpoint under a consistent protocol
- plots comparing static errors with RDFs, diffusion, energy drift, and NPT density where available
- a concise written interpretation of whether static errors predict water MD observables

The strongest outcome would be a clear non-monotonic or threshold relationship showing that lower force error does not automatically imply better liquid-water MD behavior.

## Risks And Mitigations

Risk: The dataset format is inconvenient for the chosen training framework.
Mitigation: Convert through ASE-compatible formats and document the conversion path.

Risk: Training MACE is too expensive on available hardware.
Mitigation: Use a smaller model, fewer configurations, shorter training, or CPU-feasible smoke tests before full training.

Risk: Checkpoints do not differ enough in validation quality.
Mitigation: Save earlier checkpoints, adjust training duration, or train small variants with controlled loss weights as a secondary design.

Risk: MD runs are unstable for weak checkpoints.
Mitigation: Treat instability as a valid result and report the failure threshold, while keeping short sanity checks before longer trajectories.

Risk: Observables are noisy.
Mitigation: Start with short, repeated trajectories and focus first on robust metrics such as RDFs and NVE drift before relying on diffusion estimates.

## Expected Project Shape

The repository should eventually contain:

- dataset download and conversion notes
- training configuration files
- scripts for static evaluation
- scripts for MD execution
- scripts or notebooks for observable analysis
- figures showing metric-to-observable relationships
- a final research note summarizing methods, results, and limitations

The first implementation plan should focus on getting one end-to-end checkpoint workflow running before broadening the benchmark.
