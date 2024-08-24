import numpy as np

def correlation_sample(data, r, n, dim_v):
    nall = data['x'].shape[0]
    prob = np.ones(nall)

    ite = data['mu1']-data['mu0']

    if r!=0.0:
        for idv in range(dim_v):
            d = np.abs(data['x'][:, -idv - 1] - np.sign(r) * ite)
            prob = prob * np.power(np.abs(r), -10 * d)
    prob = prob / np.sum(prob)
    idx = np.random.choice(range(nall), n, p=prob)
    x = data['x'][idx, :]
    t = data['t'][idx]
    mu0 = data['mu0'][idx]
    mu1 = data['mu1'][idx]

    # continuous y
    y0_cont = mu0 + np.random.normal(loc=0., scale=.1, size=n)
    y1_cont = mu1 + np.random.normal(loc=0., scale=.1, size=n)

    yf_cont, ycf_cont = np.zeros(n), np.zeros(n)
    yf_cont[t>0], yf_cont[t<1] = y1_cont[t>0], y0_cont[t<1]
    ycf_cont[t>0], ycf_cont[t<1] = y0_cont[t>0], y1_cont[t<1]

    # binary y
    median_0 = np.median(mu0)
    median_1 = np.median(mu1)
    mu0[mu0 >= median_0] = 1.
    mu0[mu0 < median_0] = 0.
    mu1[mu1 < median_1] = 0.
    mu1[mu1 >= median_1] = 1.

    yf_bin, ycf_bin = np.zeros(n), np.zeros(n)
    yf_bin[t>0], yf_bin[t<1] = mu1[t>0], mu0[t<1]
    ycf_bin[t>0], ycf_bin[t<1] = mu0[t>0], mu1[t<1]

    # return
    biny_dict = {'x':x, 't':t, 'yf':yf_bin, 'ycf':ycf_bin, 'mu0':mu0, 'mu1':mu1}
    conty_dict = {'x':x, 't':t, 'yf':yf_cont, 'ycf':ycf_cont, 'mu0':y0_cont, 'mu1':y1_cont}

    return biny_dict, conty_dict


def get_var_df(df, var):
    var_cols = [c for c in df.columns if c.startswith(var)] 
    return df[var_cols].to_numpy()
