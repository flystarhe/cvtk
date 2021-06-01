import pandas as pd


def loads(dfs, how=None, cols=None):
    if isinstance(dfs, str):
        dfs = [pd.read_csv(f) for f in dfs.split(",") if f.strip()]

    if isinstance(dfs, list):
        data, n = pd.concat(dfs, ignore_index=True), len(dfs)
    else:
        raise Exception(f"{type(dfs)} not supported")

    assert isinstance(data, pd.DataFrame)

    if cols is None:
        cols = data.columns.tolist()
    else:
        data = data[cols]

    if how == "or":
        data = data.drop_duplicates()
    elif how == "and":
        rows = [k for k, v in data.groupby(cols) if v.shape[0] >= n]
        data = pd.DataFrame(rows, columns=cols)

    return data
