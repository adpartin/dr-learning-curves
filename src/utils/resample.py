import numpy as np
import pandas as pd


def flatten_dist(df, n, score_name):
    """ Resample the input df to obtain n samples. Resample such that df_out
    has a more flatten (uniform) distribution of values in col score_name.
    Args:
        df: input df
        n: number of samples to sample from df
        score_name: col name of the vector to resample
    Returns:
        df_out: resampled df
    """
    df = df.sort_values(score_name, ascending=False).reset_index(drop=True)

    # Create bins
    n_bins = 100
    # bins = np.linspace(0, df[score_name].max(), n_bins+1)
    bins = np.linspace(df[score_name].min(), df[score_name].max(), n_bins+1)
    df['bin'] = pd.cut(df[score_name].values, bins, precision=5, include_lowest=True)
    # print(df['bin'].nunique())

    # Create a counter for the bins and sort
    df['count'] = 1
    bn = df.groupby(['bin']).agg({'count': sum}).reset_index()
    bn = bn.sort_values('count').reset_index(drop=True)

    # These vars are adjusted
    n_bins = len(bn)  # the actual number of bins
    n_per_bin = int(n / n_bins)  # samples per bin

    n_ = n  # n is adjusted as we iterate over bins
    indices = []
    for r in range(bn.shape[0]):
        b = bn.loc[r, 'bin']    # the bin (interval)
        c = bn.loc[r, 'count']  # count of samples in the bin
        if c == 0:
            n_ = n_  # same since we didn't collect samples
            n_bins = n_bins - 1  # less bins by 1
            n_per_bin = int(n_ / n_bins)  # update ratio

        elif n_per_bin > c:
            idx = df.index.values[ df['bin'] == b ]
            assert len(set(indices).intersection(set(idx))) == 0, 'Indices overlap'
            indices.extend(idx)  # collect all samples in this bin

            n_ = n_ - len(idx)   # less samples left
            n_bins = n_bins - 1  # less bins by 1
            n_per_bin = int(n_ / n_bins)  # update ratio

        else:
            idx = df.index.values[ df['bin'] == b ]
            idx = np.random.choice(idx, size=n_per_bin, replace=False)  # sample indices
            assert len(set(indices).intersection(set(idx))) == 0, 'Indices overlap'
            indices.extend(idx)

    # Due to casting of int(n_/n_bins), we might need to add more samples
    if len(indices) < n:
        idx = list(set(df.index.values).difference(set(indices)))  # extract unused indices
        idx = np.random.choice(idx, size=n - len(indices), replace=False)  # sample indices
        indices.extend(idx)
        assert len(indices) == len(np.unique(indices)), 'Indices overlap'

    # Finally, sample the 'indices'
    df_out = df.loc[indices, :].reset_index(drop=True)
    df_out = df_out.drop(columns=['bin', 'count'])  # drop the utility columns

    return df_out
