import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize


def drawdown(return_series: pd.Series):
    """Takes a time series of returns for an asset.
    Returns a DataFrame with columns for:
    * wealth index
    * previous peaks
    * percentage drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Previous Peaks": previous_peaks,
        "Drawdown": drawdowns
    })


def skew_kurt(r, factor):
    demeaned_r = r - r.mean()
    # Use population std, so set ddof to 0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**factor).mean()
    return exp/sigma_r**factor


def skewness(r):
    """Alternative to scipy.stats.kurtosis"""
    return skew_kurt(r, 3)


def kurtosis(r):
    return skew_kurt(r, 4)


def is_normal(r, level=0.01):
    """Applies the Jaques-Bera test to determine if a series is normal or not.
       Test is applied at the 1% level (default) of confidence by default.
       Returns True if the hypothesis of normal distribution is accepted, False otherwise.
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def semideviation(r):
    """
    Returns the semi-deviation aka negative semideviation for a series.
    r a pandas Series or DataFrame
    """
    is_negative = r < 0  # predicate/mask to select negative entries
    return r[is_negative].std(ddof=0)


def var_historic(r, level=5):
    """
    Returns the historic Value at Risk a specified level, ie. returns the 
    number of such that "level" percent of the returns fall below that number, 
    and the (100-level) percent are above.
    """
    if isinstance(r, pd.DataFrame):
        # Runs it on every column (series)
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected t to be a pandas Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a pandas Series or DataFrame.
    If the parameter `modified` is True, then the function modifies the VaR
    using the Cornish-Fisher modification.
    """
    # compute the Z score assuming it was Gaussian (the distance from the mean distr)
    z = stats.norm.ppf(level/100)

    if modified:
        # modify the z score based on skew and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
             )
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    """
    Computes the Conditional Value at Risk (CVAR) for a Series or DataFrame
    """
    if isinstance(r, pd.Series):
        # all returns below historic VAR:
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Pandas Series or DataFrame.")


def annualized_returns(r, periods_per_year):
    """
    Annualizes the the returns for a series `r`.
    TODO: Infer the periods per year from r
    """
    compounded_returns = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_returns**(periods_per_year/n_periods) - 1


def annualized_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set `r` of returns.
    TODO: We should infer the periods per year from r
    """
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Compute the annualized Sharpe ration for a set of returns
    """
    # convert the annual risk free rate to a rate per period:
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year) - 1
    excess_return = r - rf_per_period
    ann_ex_ret = annualized_returns(excess_return, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def portfolio_return(weights, returns):
    """
    Weights --> Return
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Weights --> Volatility
    """
    return (weights.T @ covmat @ weights)**0.5


def minimize_vol(target_return, er, cov):
    """
    target_returns --> weights which minimalizes the volatility for a give target_return
    """
    n = er.shape[0]
    initial_weights = np.repeat(1/n, n)  # Equally distr. weights
    bounds = ((0.0, 1.0),)*n            # n bounds of (0,1) tuples
    constraint_return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    constraint_weight_sum_is_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, initial_weights, args=(cov,), method="SLSQP", options={
                       'disp': False}, constraints=(constraint_return_is_target, constraint_weight_sum_is_one), bounds=bounds)
    return results.x


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights to minimize volatility
    """
    target_returns = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target, er, cov) for target in target_returns]
    return weights


def msr(riskfree_rate, er, cov):
    """
    riskfree_rate, er, cov --> weights which give us the max. Sharpe ratio

    er: expected returns
    cov: covariance metrics for assets
    """
    n = er.shape[0]
    initial_weights = np.repeat(1/n, n)  # Equally distr. weights
    bounds = ((0.0, 1.0),)*n            # n bounds of (0,1) tuples
    constraint_weight_sum_is_one = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the inverse of the Sharpe ratio given:
        * weights: allocation of the assets
        """
        r = portfolio_return(weights, er)
        v = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/v

    results = minimize(neg_sharpe_ratio, initial_weights, args=(riskfree_rate, er, cov,), method="SLSQP", options={
                       'disp': False}, constraints=(constraint_weight_sum_is_one), bounds=bounds)
    return results.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Variance portfolio
    for a given covariance matrix (cov)
    """
    n = cov.shape[0]
    return msr(0.0, np.repeat(1, n), cov)


def plot_ef2(n_points, er, cov):
    """
    Plots the efficient frontier for 2 assets
    """
    if er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2 asset frontiers.")

    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    returns = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    efficientFrontier = pd.DataFrame({"Returns": returns, "Volatility": vols})
    return efficientFrontier.plot.line(x="Volatility", y="Returns", style=".-")


def plot_ef(n_points, er, cov, show_cml=False, style=".-", riskfree_rate=0.0, show_ew=False, show_gmv=False):
    """
    Plots the n-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=".-")
    ax.set_xlim(left=0)

    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        v_ew = portfolio_vol(w_ew, cov)
        # Add EW
        ax.plot([v_ew], [r_ew], color="goldenrod", marker="o", markersize=12)

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        v_gmv = portfolio_vol(w_gmv, cov)
        # Add GMV
        ax.plot([v_gmv], [r_gmv], color="midnightblue",
                marker="o", markersize=12)

    if show_cml:
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        v_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, v_msr]  # Range from zero to v_msr
        cml_y = [riskfree_rate, r_msr]  # Range from riskfree_rate to r_msr
        ax.plot(cml_x, cml_y, color="green", marker="o",
                markersize=10, linewidth=2, linestyle="dashed")

    return ax


# Utility functions (get rid of them later)
# -----------------------------------------


def get_ffme_returns():
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[["Lo 10", "Hi 10"]]
    rets.columns = ["SmallCap", "LargeCap"]
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets


def get_hfi_returns():
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period("M")
    hfi.columns = hfi.columns.str.strip()
    return hfi


def get_ind_returns():
    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0,
                      index_col=0, parse_dates=True)
    ind = ind/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind
