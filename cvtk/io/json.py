try:
    import simplejson as json
except ImportError:
    import json


def json_dumps(obj, **kw):
    return json.dumps(obj, **kw)


def json_loads(s, **kw):
    return json.loads(s, **kw)
