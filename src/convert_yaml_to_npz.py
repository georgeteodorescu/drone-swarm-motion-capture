import yaml
import numpy as np


# functie pentru a inlocui None cu np.nan
def replace_none_with_nan(element):
    if isinstance(element, list):
        return [replace_none_with_nan(sub_element) for sub_element in element]
    elif element is None:
        return np.nan
    return element


# incarcare fisier YAML
with open('tracked_points.yaml', 'r') as file:
    data = yaml.safe_load(file)

# inlocuire None cu np.nan in date
data_with_nan = replace_none_with_nan(data)

# convertire catre vector NumPy
np_array = np.array(data_with_nan, dtype=object)

# utilizare np.savez_compressed penru salvarea fisierului intr-un format comprimat
np.savez_compressed('output.npz', tracked_points=np_array)
