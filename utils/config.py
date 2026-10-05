"""utils/config.py — dict with attribute access for nested YAML configs."""


class AttrDict(dict):
    """A dict whose keys are also accessible as attributes (nested-safe)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def to_attr_dict(d):
    """Recursively convert plain dicts into :class:`AttrDict`."""
    if isinstance(d, dict):
        return AttrDict({k: to_attr_dict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_attr_dict(v) for v in d]
    return d