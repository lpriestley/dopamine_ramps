"""
Main entry point for running experiments and generating figures.
"""
import numpy as np
import experiments as exp

# Global random seed
np.random.seed(18)

RUN_DEMO = True
RUN_COMPARE_ARCHITECTURES = True
RUN_GURU = True
RUN_KIM = False
RUN_KRAUSZ = True
RUN_MIKHAEL = False

if RUN_DEMO:
    exp.dual_process_demo()

if RUN_COMPARE_ARCHITECTURES:
    exp.compare_architectures()

if RUN_GURU:
    exp.guru_track()

if RUN_KIM:
    exp.kim_distance()
    exp.kim_location()
    exp.kim_speed()

if RUN_KRAUSZ:
    exp.krausz_grid()

if RUN_MIKHAEL:
    exp.compare_uncertainty_assumptions()
    exp.mikhael_track()
