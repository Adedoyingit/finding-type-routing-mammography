"""Finding-type-aware ensemble routing for mammographic lesion classification.

Subpackages and modules
-----------------------
config      Training recipe, backbones and the twelve ensemble conditions.
data        Split loading, image transforms, datasets and samplers.
models      Single-backbone models and ensemble architectures (including AFAGE).
training    Training loop, checkpointing and multi-run execution.
inference   Prediction export and re-scoring of checkpoints.
stats       AUC, partial AUC, operating points and the seed x patient bootstrap.
analysis    The analyses reported in the paper.
figures     Figures 1-7 and Table 2.
"""
__version__ = "1.1.0"
