#https://stackoverflow.com/a/49802489/5917194
import functools
import warnings

def deprecated_alias(**aliases):
    def deco(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)
        return wrapper
    return deco

def rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise TypeError('{} received both {} and {}'.format(
                    func_name, alias, new))
            warnings.warn('{} is deprecated; use {}'.format(alias, new),
                          DeprecationWarning)
            kwargs[new] = kwargs.pop(alias)

def _test_raised_deprecation_warning(w):
    assert issubclass(w[-1].category, DeprecationWarning)
    assert "deprecated" in str(w[-1].message)
    w.pop()
