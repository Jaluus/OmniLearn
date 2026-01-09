import os
import pickle


def load_pickle(folder, f):
    file_name = os.path.join(folder, "histories", f.replace(".weights.h5", ".pkl"))
    with open(file_name, "rb") as file_pi:
        history_dict = pickle.load(file_pi)
    return history_dict

