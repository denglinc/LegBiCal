# Visualization architecture

Drake owns playback time and routes one immutable scientific frame contract to
Meshcat and MuJoCo. Rendering code never owns estimator state, and the MuJoCo
path uses kinematic forwarding rather than simulation stepping.

## Drake ownership

`build_visualization_diagram()` constructs a real `pydrake.systems.framework`
`Diagram`. `MotionForcePlaybackSystem` maps Context time to a transition index
and publishes `VisualizationFrame` through an abstract port. Meshcat and
MuJoCo sink LeafSystems consume that same frame and call their backend
renderers. Either or both sinks can be connected without changing the sequence
or time owner.

## Modules

| Module | Responsibility |
|---|---|
| [`types.py`](types.py) | Immutable style, contact, sequence, and frame types |
| [`sequence.py`](sequence.py) | Validates saved solutions and constructs backend-independent sequences |
| [`playback_system.py`](playback_system.py) | Drake time-to-frame `LeafSystem` |
| [`diagram.py`](diagram.py) | Canonical `DiagramBuilder` composition for one or both sinks |
| [`meshcat_system.py`](meshcat_system.py) | Meshcat renderer, checker floor, animation export, playback-rate controls, and publish sink |
| [`mujoco_system.py`](mujoco_system.py) | Drake MuJoCo publish sink |
| [`mujoco_renderer.py`](mujoco_renderer.py) | Offscreen frame renderer and passive interactive viewer |
| [`force_geometry.py`](force_geometry.py) | Shared force arrows, support polygon, center of pressure, and friction metrics |
| [`recording.py`](recording.py) | Deterministic sampling and numerical artifact sidecars |

The Meshcat HTML defaults to `0.5x`; its injected controls switch between
`0.5x` and `1x`, and `?speed=1` selects normal speed on load. Both viewers use
the PRIME experiment checker palette. Force geometry and colors are defined by
the shared `VisualizationStyle` loaded from
[`configs/visualization/default.yaml`](../../../configs/visualization/default.yaml).

Return to the [`g1cal` package](../README.md).
