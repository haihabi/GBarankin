def input_validation(param_name, **kwargs):
    if kwargs.get(param_name) is None:
        raise Exception(f"Cant find param named:{param_name}")
    base_param = kwargs.get(param_name)
    if len(base_param.shape) != 2:
        raise Exception("Param shape mismatch")
    if base_param.shape[0] != 1:
        raise Exception("aa")
