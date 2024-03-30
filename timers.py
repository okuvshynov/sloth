import time
from contextlib import contextmanager
import fewlines
import logging

timer_context = {}

@contextmanager
def AdditiveTimer(name):
    start_time = time.monotonic_ns()
    try:
        yield
    finally:
        end_time = time.monotonic_ns()
        elapsed_time = end_time - start_time
        # Append the measurement to the list of times for this name
        if name not in timer_context:
            timer_context[name] = 0.0
        timer_context[name] += (elapsed_time / 1000000)
        #print(f"Timer '{name}': {elapsed_time} ns")

def log_metrics():
    for l in fewlines.bar_histograms(timer_context, bins=40, custom_range=(0, 500)):
        logging.info(l)