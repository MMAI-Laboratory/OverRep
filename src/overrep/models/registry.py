_model_registry = {}


def register_model(fn):
    name = fn.__name__
    _model_registry[name] = fn
    return fn


def create_model(name, *args, **kwargs):
    if name not in _model_registry:
        raise ValueError(f"Model {name} is not registered.")
    return _model_registry[name](*args, **kwargs)


def list_models():
    return list(_model_registry.keys())
