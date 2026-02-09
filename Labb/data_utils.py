import numpy as np

def load_data(path, delimiter=",", skip_header=True):
    with open(path, "r", encoding="utf-8") as f:
        data = f.readlines()
    return data

