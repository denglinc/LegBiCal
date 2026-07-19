# Configuration subpackage

[`options.m`](options.m) is the single declaration point for dataset windows,
state dimensions, covariance layout, kinematic bounds, Fatrop settings, loss
weights, and upper-level defaults.

Keeping these values in one immutable options structure ensures that graph
construction, derivative generation, and every optimizer use identical
dimensions and constraints.

Return to the [`legbical` package](../README.md).
