import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple

@njit
def _ewm_std(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Exponential Weighted Moving Std (EWMA Std)."""
    n = len(data)
    alpha = 2 / (period + 1)
    mean = np.empty(n, dtype=np.float64)
    var = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)
    mean[0] = data[0]
    var[0] = 0.0
    std[0] = 0.0
    for i in range(1, n):
        mean[i] = alpha * data[i] + (1 - alpha) * mean[i - 1]
        var[i] = (1 - alpha) * (var[i - 1] + alpha * (data[i] - mean[i - 1]) ** 2)
        std[i] = np.sqrt(var[i])
    return std

@njit
def _sma_std(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute simple rolling standard deviation (SMA Std)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        window = data[i - period:i]
        result[i] = np.std(window)
    return result

def rolling_std(data: NDArray[np.float64], period: int, method: str = "EWMA") -> NDArray[np.float64]:
    """
    General rolling standard deviation wrapper.

    Args:
        data: Price series
        period: Window size
        method: 'EWMA' or 'SMA'

    Returns:
        Standard deviation series.
    """
    method = method.upper()
    if method == "EWMA":
        return _ewm_std(data, period)
    elif method == "SMA":
        return _sma_std(data, period)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'EWMA' or 'SMA'.")

@njit
def _ema(
    data: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Exponential Moving Average."""
    n = len(data)
    alpha = 2 / (period + 1)
    result = np.empty(n, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

@njit
def _sma(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Simple Moving Average (SMA)."""
    n = len(data)
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma

def moving_average(
    data: NDArray[np.float64],
    period: int,
    method: str = "EMA"
) -> NDArray[np.float64]:
    """Moving Average (MA): Simple, EMA, SMA."""
    if method.lower() == "ema":
        return _ema(data, period)
    elif method.lower() == "sma":
        return _sma(data, period)
    else:
        raise ValueError("Invalid MA type. Choose 'EMA', 'SMA'.")

@njit
def _macd(
    data: NDArray[np.float64],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute MACD indicator.

    Returns:
        - macd_line: EMA(fast) - EMA(slow)
        - signal_line: EMA(macd_line, signal_period)
        - histogram: macd_line - signal_line
    """
    fast_ema = _ema(data, fast_period)
    slow_ema = _ema(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def _calculate_bollinger_bands(
    close: NDArray[np.float64],
    period: int,
    std_multiplier: float = 2.0,
    ma_method: str = "EMA",
    std_type: str = "EWMA"
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Bollinger Bands (upper, mid, lower)."""
    std = rolling_std(close, period, std_type)
    mid = moving_average(close, period, ma_method)
    upper = mid + std_multiplier * std
    lower = mid - std_multiplier * std
    return upper, mid, lower

@njit
def _adx(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Average Directional Index (ADX)."""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

    atr = np.zeros(n, dtype=np.float64)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)

    plus_dm_smooth = np.zeros(n, dtype=np.float64)
    minus_dm_smooth = np.zeros(n, dtype=np.float64)
    plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])

    for i in range(period + 1, n):
        plus_dm_smooth[i] = (plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i])
        minus_dm_smooth[i] = (minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i])

        if atr[i] > 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / atr[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / atr[i])

        di_diff = abs(plus_di[i] - minus_di[i])
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * di_diff / di_sum

    adx = np.full(n, np.nan)
    adx[period * 2] = np.mean(dx[period+1:period*2+1])
    for i in range(period * 2 + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


@njit
def _rsi(
    close: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Relative Strength Index (RSI)."""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan, dtype=np.float64)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    rs = np.zeros(n, dtype=np.float64)
    rsi = np.full(n, np.nan, dtype=np.float64)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
        if avg_loss[i] == 0:
            rs[i] = np.inf
            rsi[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    return rsi

@njit
def _average_true_range(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int
) -> NDArray[np.float64]:
    """Compute Average True Range (ATR)."""
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = np.abs(high[i] - close[i - 1])
        lc = np.abs(low[i] - close[i - 1])
        tr = max(hl, hc, lc)
        if i == period:
            total = 0.0
            for j in range(1, period + 1):
                hl = high[j] - low[j]
                hc = np.abs(high[j] - close[j - 1])
                lc = np.abs(low[j] - close[j - 1])
                total += max(hl, hc, lc)
            atr[i] = total / period
        elif i > period:
            atr[i] = (atr[i - 1] * (period - 1) + tr) / period
    return atr

def _trend_detect(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int,
    threshold: float = 0.4
) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Detect trend based on ATR/std ratio."""
    atr_vals = _average_true_range(high, low, close, period)
    std_vals = moving_average(close, period, "EMA")
    ratio = atr_vals / std_vals
    n = len(ratio)
    mean_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        mean_ratio[i] = np.mean(ratio[i - 3:i])
    trending = np.where(mean_ratio <= threshold, 1, 0)
    return mean_ratio, trending

@njit
def _calculate_volatility_normalize(
    data: NDArray[np.float64],
    timeperiod: int = 14
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize volatility from log returns."""
    n = len(data)
    log_returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        log_returns[i] = np.log(data[i] / data[i - 1])
    vol = np.zeros(n, dtype=np.float64)
    for i in range(timeperiod - 1, n):
        r_slice = log_returns[i - timeperiod + 1:i + 1]
        std = np.std(r_slice)
        vol[i] = std * np.sqrt(timeperiod)
    recent_vol = vol[-timeperiod:]
    vmin = np.min(recent_vol)
    vmax = np.max(recent_vol)
    normalized_vol = np.zeros_like(recent_vol)
    if vmax > vmin:
        for i in range(timeperiod):
            normalized_vol[i] = (recent_vol[i] - vmin) / (vmax - vmin)
    return normalized_vol, recent_vol

@njit
def _calculate_kelly_criterion(
    data: NDArray[np.float64],
    slow_period: int = 60,
    fast_period: int = 10,
    kelly_fraction: float = 0.5
) -> Tuple[NDArray[np.float64], float]:
    """
    Compute Kelly position sizing based on fast/slow volatility ratio.

    Returns:
        - kelly_criterion: full array of sizing values
        - recent_criterion: latest value as float
    """
    n = len(data)
    log_returns = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        log_returns[i] = np.log(data[i] / data[i - 1])
    
    vol_f = np.zeros(n, dtype=np.float64)
    vol_s = np.zeros(n, dtype=np.float64)
    
    for i in range(slow_period - 1, n):
        r_slice = log_returns[i - slow_period + 1:i + 1]
        std = np.std(r_slice)
        vol_f[i] = std * np.sqrt(slow_period)
    
    for i in range(fast_period - 1, n):
        r_slice = log_returns[i - fast_period + 1:i + 1]
        std = np.std(r_slice)
        vol_s[i] = std * np.sqrt(fast_period)
    
    kelly_criterion = vol_s / vol_f * kelly_fraction

    return kelly_criterion

@njit
def estimate_q_r(price: np.ndarray, window: int = 60) -> Tuple[float, float]:
    n = len(price)
    if n < window + 2:
        # Not enough data, fallback to simple variance of returns for both
        log_returns = np.diff(np.log(price))
        var_lr = np.var(log_returns) if len(log_returns) > 0 else 1e-6
        return var_lr, var_lr

    log_returns = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        log_returns[i - 1] = np.log(price[i] / price[i - 1])

    # Observation noise R: variance of last 'window' log returns
    mean_lr = 0.0
    for i in range(n - window - 1, n - 1):
        mean_lr += log_returns[i]
    mean_lr /= window

    R = 0.0
    for i in range(n - window - 1, n - 1):
        R += (log_returns[i] - mean_lr) ** 2
    R /= window

    # Process noise Q: variance of diff of log returns (acceleration)
    process_noise_len = n - 2
    process_noise = np.empty(process_noise_len, dtype=np.float64)
    for i in range(1, n - 1):
        process_noise[i - 1] = log_returns[i] - log_returns[i - 1]

    mean_pn = 0.0
    for i in range(process_noise_len - window, process_noise_len):
        mean_pn += process_noise[i]
    mean_pn /= window

    Q = 0.0
    for i in range(process_noise_len - window, process_noise_len):
        Q += (process_noise[i] - mean_pn) ** 2
    Q /= window

    # Prevent zero variance (which breaks Kalman)
    if Q == 0.0:
        Q = 1e-8
    if R == 0.0:
        R = 1e-8

    return Q, R

@njit
def _kalman_filter_mean(
    data: NDArray[np.float64],
    process_variance: float,
    measurement_variance: float,
    adjust_factor: float = 1.0
) -> NDArray[np.float64]:
    """
    Kalman Filter to estimate dynamic mean of a time series.
    
    Args:
        data: Input price series.
        process_variance: Q - variance of the process (smoother = smaller).
        measurement_variance: R - variance of the observations (noisier = larger).
    
    Returns:
        Filtered series as dynamic mean estimates.
    """
    n = len(data)
    estimated = np.empty(n, dtype=np.float64)
    estimate = data[0]  # initial estimate
    error_estimate = 1.0  # initial error estimate

    for t in range(n):
        # Prediction update
        error_estimate += process_variance * adjust_factor

        # Measurement update (Kalman gain)
        kalman_gain = error_estimate / (error_estimate + measurement_variance)
        estimate = estimate + kalman_gain * (data[t] - estimate)
        error_estimate = (1 - kalman_gain) * error_estimate

        estimated[t] = estimate

    return estimated

@njit
def _kalman_filter_beta(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    process_variance: float = 1e-5,
    measurement_variance: float = 1e-3,
    adjust_factor: float = 1.0,
    init_beta: float = 1.0
) -> NDArray[np.float64]:
    """
    Kalman Filter to estimate time-varying beta in y = beta * x.

    Args:
        y: Dependent variable time series (n,).
        x: Independent variable time series (n,).
        process_variance: Q - variance of beta evolution (smoother = smaller).
        measurement_variance: R - observation noise variance.
        adjust_factor: Multiplier on process variance (Q) to control adaptivity.
        init_beta: Initial beta estimate.

    Returns:
        Estimated beta series of length n.
    """
    n = len(y)
    beta_est = np.empty(n, dtype=np.float64)

    # Initial state
    beta = init_beta
    P = 1.0  # initial estimation error covariance

    for t in range(n):
        # Prediction step
        P = P + process_variance * adjust_factor

        # Measurement update
        if x[t] != 0.0:
            H = x[t]
            R = measurement_variance
            # Kalman Gain
            K = (P * H) / (H * P * H + R)
            # Update beta
            beta = beta + K * (y[t] - beta * H)
            # Update covariance
            P = (1 - K * H) * P

        beta_est[t] = beta

    return beta_est

def kalman_filter_mean(
    data: np.ndarray,
    process_variance: float = -1.0,
    measurement_variance: float = -1.0,
    adjust_factor: float = 1.0
) -> np.ndarray:
    """
    Wrapper for _kalman_filter_mean that estimates Q and R if not provided.
    """
    if process_variance < 0 or measurement_variance < 0:
        process_variance, measurement_variance = estimate_q_r(data)
    return _kalman_filter_mean(data, process_variance, measurement_variance, adjust_factor)

def estimate_q_r_beta(y: np.ndarray, x: np.ndarray):
    """
    Estimate Q and R for Kalman filter beta.
    """
    # OLS beta
    beta_ols = np.sum(x * y) / np.sum(x * x)
    residuals = y - beta_ols * x

    # Measurement noise variance
    R = np.var(residuals)

    # Process noise variance (small fraction of R, adaptivity factor)
    Q = 0.01 * R if R > 0 else 1e-6

    return Q, R

def kalman_filter_beta(
    y: np.ndarray,
    x: np.ndarray,
    process_variance: float = -1.0,
    measurement_variance: float = -1.0,
    adjust_factor: float = 1.0,
    init_beta: float = 1.0
) -> np.ndarray:
    """
    Wrapper for _kalman_filter_beta that estimates Q and R if not provided.
    """
    if process_variance < 0 or measurement_variance < 0:
        process_variance, measurement_variance = estimate_q_r_beta(y, x)
    return _kalman_filter_beta(y, x, process_variance, measurement_variance, adjust_factor, init_beta)

@njit
def _z_score(
    series: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    epsilon: float = 1e-8
) -> NDArray[np.float64]:
    """
    Compute Z-score: (series - mean) / std with epsilon to avoid division by zero.
    
    Args:
        series: Input data (e.g., price).
        mean: Smoothed mean (e.g., Kalman, EMA).
        std: Volatility estimate (e.g., rolling/EWM std).
        epsilon: Small constant to prevent divide-by-zero.
    
    Returns:
        Z-score array.
    """
    n = len(series)
    z = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if std[i] > epsilon:
            z[i] = (series[i] - mean[i]) / std[i]

    return z

@njit
def _cross_value(series: NDArray[np.float64], value: float) -> NDArray[np.float64]:
    """
    Detect cross-over and cross-under events of a series vs. a constant value.

    Args:
        series: Input time series (e.g., price or indicator).
        value: Constant threshold to check for crossing.

    Returns:
        NDArray[np.float64]: Array with non-zero values only at cross points:
            +diff → upward cross (series crosses above value)
            -diff → downward cross (series crosses below value)
            0     → no cross
    """
    n = len(series)
    output = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        prev = series[i - 1]
        curr = series[i]

        if prev < value and curr > value:
            output[i] = curr - value  # upward cross
        elif prev > value and curr < value:
            output[i] = curr - value  # downward cross (negative)
        else:
            output[i] = 0.0

    return output

@njit
def _cross(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
    """
    Detect cross-over events between two series.

    Args:
        series_a: First input series (e.g., price).
        series_b: Second input series (e.g., moving average).

    Returns:
        np.ndarray: Array where:
            +1 = cross over (a crosses above b),
            -1 = cross under (a crosses below b),
             0 = no cross.
    """
    n = len(series_a)
    crosses = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        prev_a = series_a[i - 1]
        prev_b = series_b[i - 1]
        curr_a = series_a[i]
        curr_b = series_b[i]

        if prev_a < prev_b and curr_a > curr_b:
            crosses[i] = 1
        elif prev_a > prev_b and curr_a < curr_b:
            crosses[i] = -1
        else:
            crosses[i] = 0

    return crosses

@njit
def _average_linkage_distance(cluster_a, cluster_b, dist):
    d = 0.0
    count = 0
    for i in range(len(cluster_a)):
        for j in range(len(cluster_b)):
            d += dist[cluster_a[i], cluster_b[j]]
            count += 1
    return d / count

@njit
def _hierarchical_clustering(corr: NDArray[np.float64],
                            n_clusters: int = 3,
                            max_dist: float = 0.5) -> NDArray[np.int64]:
    """
    Numba-safe hierarchical clustering using average linkage, with max distance cutoff.

    Args:
        corr: Correlation matrix (NxN).
        n_clusters: Target number of clusters (upper bound).
        max_dist: Do not merge clusters if average distance > max_dist.

    Returns:
        cluster_labels: Array of cluster indices for each asset.
    """
    n = corr.shape[0]
    dist = np.sqrt(0.5 * (1.0 - corr))  # Convert correlation to distance
    
    clusters = List()
    for i in range(n):
        clusters.append(np.array([i], dtype=np.int64))

    while len(clusters) > n_clusters:
        min_dist = 1e9
        merge_a, merge_b = -1, -1

        # Find closest pair of clusters
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                d = _average_linkage_distance(clusters[i], clusters[j], dist)
                if d < min_dist:
                    min_dist = d
                    merge_a, merge_b = i, j

        # Stop merging if best pair is still far apart
        if min_dist > max_dist:
            break

        # Merge clusters
        merged = np.concatenate((clusters[merge_a], clusters[merge_b]))
        if merge_a > merge_b:
            del clusters[merge_a]
            del clusters[merge_b]
        else:
            del clusters[merge_b]
            del clusters[merge_a]
        clusters.append(merged)

    # Assign labels
    labels = np.empty(n, dtype=np.int64)
    for cluster_id in range(len(clusters)):
        for k in range(len(clusters[cluster_id])):
            labels[clusters[cluster_id][k]] = cluster_id

    return labels


@njit
def _cointegration(
    x: NDArray[np.float64],
    y: NDArray[np.float64]
) -> Tuple[float, float]:
    """
    Compute Engle-Granger cointegration for two price series.
    Returns:
        beta: Hedge ratio (OLS slope)
        t_stat: ADF-like t-statistic for stationarity of residuals
    """
    n = len(x)
    mean_x = 0.0
    mean_y = 0.0
    
    # Compute means
    for i in range(n):
        mean_x += x[i]
        mean_y += y[i]
    mean_x /= n
    mean_y /= n
    
    # Compute beta (OLS slope)
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        num += dx * dy
        den += dx * dx
    beta = num / den if den != 0 else 0.0
    
    # Residuals
    residuals = np.empty(n, dtype=np.float64)
    for i in range(n):
        residuals[i] = y[i] - beta * x[i]
    
    # ADF-like test statistic
    # Δres = residuals[1:] - residuals[:-1]
    res_diff = np.empty(n - 1, dtype=np.float64)
    for i in range(1, n):
        res_diff[i - 1] = residuals[i] - residuals[i - 1]
    
    # Regression Δres = alpha * res[-1] + error
    num_adf = 0.0
    den_adf = 0.0
    for i in range(1, n):
        num_adf += residuals[i - 1] * res_diff[i - 1]
        den_adf += residuals[i - 1] * residuals[i - 1]
    alpha = num_adf / den_adf if den_adf != 0 else 0.0
    
    # Compute standard error
    err_sum = 0.0
    for i in range(1, n):
        pred = alpha * residuals[i - 1]
        err = res_diff[i - 1] - pred
        err_sum += err * err
    sigma2 = err_sum / (n - 2)
    se = np.sqrt(sigma2 / den_adf) if den_adf != 0 else 0.0
    
    t_stat = alpha / se if se != 0 else 0.0
    
    return beta, t_stat

class Indicator:
    macd = staticmethod(_macd)
    adx = staticmethod(_adx)
    average_true_range = staticmethod(_average_true_range)
    rolling_std = staticmethod(rolling_std)
    moving_average = staticmethod(moving_average)
    bollinger_bands = staticmethod(_calculate_bollinger_bands)
    relative_strength_index = staticmethod(_rsi)
    volatility_normalize = staticmethod(_calculate_volatility_normalize)
    kelly_criterion = staticmethod(_calculate_kelly_criterion)
    kalman_filter_mean = staticmethod(kalman_filter_mean)    # wrapper for external use
    z_score = staticmethod(_z_score)
    cross = staticmethod(_cross)
    cross_value = staticmethod(_cross_value)
    trend_detect = staticmethod(_trend_detect)
    cointegration = staticmethod(_cointegration)
    kalman_filter_beta = staticmethod(kalman_filter_beta)
    hierarchical_clustering = staticmethod(_hierarchical_clustering)


import numpy as np
from numba import njit
from numpy.typing import NDArray

# -------------------------
# 1. PCA weights
# -------------------------
@njit
def pca_basket_weights(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    T, N = returns.shape
    mean_ret = np.zeros(N)
    for i in range(N):
        s = 0.0
        for t in range(T):
            s += returns[t, i]
        mean_ret[i] = s / T

    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for t in range(T):
                s += (returns[t, i] - mean_ret[i]) * (returns[t, j] - mean_ret[j])
            cov[i, j] = s / (T - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_max = np.argmax(eigvals)
    weights = eigvecs[:, idx_max]

    sgn = 0.0
    for i in range(N):
        sgn += weights[i]
    if sgn < 0:
        weights = -weights

    abs_sum = 0.0
    for i in range(N):
        abs_sum += abs(weights[i])
    if abs_sum > 0:
        weights /= abs_sum

    return weights

# -------------------------
# 2. Basket returns
# -------------------------
@njit
def basket_returns(returns: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    T = returns.shape[0]
    basket_ret = np.zeros(T)
    for t in range(T):
        for i in range(weights.shape[0]):
            basket_ret[t] += returns[t, i] * weights[i]
    return basket_ret

# -------------------------
# 4. Compute residuals and Z-score
# -------------------------
@njit
def residual_zscore(asset: NDArray[np.float64],
                    basket: NDArray[np.float64],
                    beta: NDArray[np.float64],
                    window: int = 50) -> NDArray[np.float64]:
    T = asset.shape[0]
    residuals = np.zeros(T)
    zscores = np.zeros(T)
    for t in range(T):
        residuals[t] = asset[t] - beta[t] * basket[t]
        if t >= window:
            mean = 0.0
            var = 0.0
            for k in range(t - window, t):
                mean += residuals[k]
            mean /= window
            for k in range(t - window, t):
                var += (residuals[k] - mean) ** 2
            var /= window
            if var > 0:
                zscores[t] = (residuals[t] - mean) / np.sqrt(var)
    return zscores

@njit
def compute_all_signals(returns: NDArray[np.float64],
                        basket_ret: NDArray[np.float64],
                        window: int = 50,
                        threshold: float = 2.0,
                        adj_factors: float = 0.1) :
    """
    Compute beta, Z-score, and trade signals for all assets in a cluster.
    
    Args:
        returns: (T, N) normalized returns of assets
        basket_ret: (T,) PCA basket return
        window: rolling window for residual std
        threshold: Z-score threshold
    
    Returns:
        beta_matrix: (T, N) time-varying betas
        z_matrix: (T, N) z-scores
        signal_matrix: (T, N) trade signals (1=long, -1=short, 0=none)
    """
    T, N = returns.shape
    beta_matrix = np.zeros((T, N))
    z_matrix = np.zeros((T, N))
    signal_matrix = np.zeros((T, N), dtype=np.int64)

    for i in range(N):
        beta = _kalman_filter_beta(returns[:, i], basket_ret, adjust_factor=adj_factors)
        z = residual_zscore(returns[:, i], basket_ret, beta, window)

        beta_matrix[:, i] = beta
        z_matrix[:, i] = z

        for t in range(T):
            if z[t] > threshold:
                signal_matrix[t, i] = -1  # Short: asset overpriced vs basket
            elif z[t] < -threshold:
                signal_matrix[t, i] = 1   # Long: asset underpriced vs basket

    return beta_matrix, z_matrix, signal_matrix

class Indicators_2:
    pca_basket_weights = staticmethod(pca_basket_weights)
    basket_returns = staticmethod(basket_returns)
    residual_zscore = staticmethod(residual_zscore)
    compute_all_signals = staticmethod(compute_all_signals)