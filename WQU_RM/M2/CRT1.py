import numpy as np, statsmodels.api as sm

def d_covar(j_ret, i_ret, q=0.05):
    j = np.asarray(j_ret, float).ravel()
    i = np.asarray(i_ret, float).ravel()
    var_j_q = np.quantile(j, q)
    X = sm.add_constant(i)
    fit = sm.QuantReg(j, X).fit(q=q)
    b0, b1 = fit.params[0], fit.params[1]
    var_i_q = np.quantile(i, q)
    covar_q = b0 + b1 * var_i_q
    med_i = np.quantile(i, 0.5)
    covar_50 = b0 + b1 * med_i
    return {"VaR_j_q": var_j_q, "CoVaR_q_j|i": covar_q, "CoVaR_50_j|i": covar_50, "Delta_CoVaR": covar_q - covar_50}

# tiny demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    i = rng.normal(0,1,2000)
    j = 0.4*i + rng.normal(0,1,2000)
    print(d_covar(j, i, q=0.05))
