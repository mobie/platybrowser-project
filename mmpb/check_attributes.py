from . import attributes


def check_attributes(name):
    if not isinstance(name, str):
        return False
    if getattr(attributes, name, None) is None:
        return False
    return True
