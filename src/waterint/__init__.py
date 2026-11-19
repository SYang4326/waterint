"""WaterInt: analysis of water/solid-interface molecular simulations.

Layers (dependencies point downwards only):
    io        trajectories -> frames
    system    SystemProfile (everything system-specific)
    prep      per-frame preparation shared by all analyses (ownership, species, H bonds, reference plane)
    analysis  the science (stream analyses consume LabelledFrames)
    pipeline  one pass over the trajectory for many analyses; v0.6 config migration
    plot      figures from saved Results
    backends  python reference + optional C++ (cpu) / CUDA (gpu) kernels
"""
__version__ = "0.7.0.dev0"


def run(config):
    """Run a v0.7 pipeline config (dict) and return {name: Result}."""
    from waterint.pipeline.run import run as _run
    return _run(config)
