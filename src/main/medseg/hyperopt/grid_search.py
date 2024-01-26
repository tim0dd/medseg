import itertools


# TODO: turn this into a class like in hyperband.py
# TODO: add support for nested dictionaries and lists
def find_grid_search_params(config: dict) -> dict:
    params = {}
    for key, value in config.items():
        # check if value is an array / list
        if hasattr(value, "__len__") and type(value) != str:
            params[key] = config[key]
    return params


# return list of dictionaries, each with a unique combination of the grid search params
def get_combinations(grid_dict: dict) -> list:
    keys, values = zip(*grid_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts
