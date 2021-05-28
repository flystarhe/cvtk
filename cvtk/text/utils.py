import re


def match(pattern, string):
    # `re.compile(r".*coco_file$")`
    if pattern.match(string):
        return True
    return False


def replace(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)
    return obj


def replace2(pattern, data, old, new):
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    def _func(k, v):
        if match(pattern, k):
            return replace(v, old, new)
        return v

    assert isinstance(data, dict), f"{type(data)} not supported"
    return {k: _func(k, v) for k, v in data.items()}
