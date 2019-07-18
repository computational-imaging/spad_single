import torch
from sacred import Experiment

ex = Experiment("dorn_weighted_hist_match")

@ex.config
def cfg():
    pass


@ex.automain
def run():
    # Load SPAD histograms

    # Run stuff
    #
