# Full-information estimator

[`full_information.py`](full_information.py) builds the B1 lower problem as a
stage-structured CasADi NLP for Fatrop.

The estimator owns state, process-noise, control, and measurement stages;
arrival/process/measurement covariances enter through the shared parameter
layout. It validates state/transition alignment, carries primal and dual warm
starts, and unpacks the optimized trajectory without changing stage order.

KKT residuals, state derivatives, and parameter derivatives are generated from
the solved graph and stored as a portable derivative bundle. The upper layer
therefore differentiates the exact lower formulation rather than maintaining a
second approximate model.

Return to the [`bilevel` package](../README.md).
