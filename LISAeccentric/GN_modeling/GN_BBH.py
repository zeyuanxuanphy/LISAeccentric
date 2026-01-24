# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import scipy.interpolate as sci_interpolate
import scipy.constants as sciconsts
from scipy.optimize import brentq
import copy
import random
import os
from scipy.interpolate import PchipInterpolator  # [新增] 用于快速插值
from scipy.integrate import quad
from numba import njit
import scipy
# ==========================================
# 1. Physical Constants Definition
# ==========================================
m_sun_sec = 1.9891e30 * sciconsts.G / np.power(sciconsts.c, 3.0)
AU_sec = sciconsts.au / sciconsts.c
pc_sec = 3.261 * sciconsts.light_year / sciconsts.c
year_sec = 365 * 24 * 3600.0
day_sec = 24 * 3600.0
CONST_G = sciconsts.G
CONST_c = sciconsts.c
CONST_pi = sciconsts.pi
CONST_au = sciconsts.au
m_sun = 1.98840987e30 * CONST_G / np.power(CONST_c, 3.0)
years = 365 * 24 * 3600.0
days = 24 * 3600
pc = 3.261 * sciconsts.light_year / CONST_c
AU = CONST_au / CONST_c
gama = 0.577215664901532860606512090082402431042159335
pi=sciconsts.pi
C_val = sciconsts.c

# ==========================================
# 2. Evolutionary Physics Functions
# ==========================================
class MergerTimeAccelerator:
    """
    预计算 Peters (1964) 积分因子的插值表。
    支持磁盘缓存：首次运行会计算并保存为 .npz 文件，之后直接读取。
    """

    def __init__(self, cache_file='merger_time_table.npz'):
        self.cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_file)

        # 尝试加载缓存
        if self._load_cache():
            pass
            #print(f"[System] Loaded merger time table from {self.cache_file}")
        else:
            print("[System] Computing merger time table (first time run)...")
            self._compute_and_save_table()
            print(f"[System] Table computed and saved to {self.cache_file}")

        # 使用 Pchip 插值 (保单调性)
        self.interpolator = PchipInterpolator(self.e_grid, self.f_vals)

    def _compute_and_save_table(self):
        # 使用对数分布采样 e，重点加密 e->1 的区域
        # x 从 0 到 5, e = 1 - 10^-x
        x_vals = np.linspace(0, 5, 2000)
        self.e_grid = 1.0 - np.power(10, -x_vals)
        self.e_grid[0] = 0.0  # 修正第一个点完全为0

        # 计算对应的无量纲因子 F(e)
        self.f_vals = np.array([self._compute_dimensionless_factor(e) for e in self.e_grid])

        # 保存到磁盘
        np.savez(self.cache_file, e_grid=self.e_grid, f_vals=self.f_vals)

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return False
        try:
            data = np.load(self.cache_file)
            self.e_grid = data['e_grid']
            self.f_vals = data['f_vals']
            return True
        except Exception as e:
            print(f"[Warning] Failed to load cache: {e}")
            return False

    def _compute_dimensionless_factor(self, e0):
        """核心积分逻辑 (与原代码一致，但只计算无量纲部分)"""
        if e0 < 1e-6: return 1.0

        term_c0 = np.power(e0, 12.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e0 ** 2, 870.0 / 2299.0)
        c0_norm = (1 - e0 ** 2) / term_c0

        def integrand(e):
            numer = np.power(e, 29.0 / 19.0) * np.power(1 + 121.0 / 304.0 * e ** 2, 1181.0 / 2299.0)
            denom = np.power(1 - e ** 2, 1.5)
            return numer / denom

        integral_val, _ = quad(integrand, 0, e0)

        # Factor = (48/19) * (c0/a)^4 * I
        return (48.0 / 19.0) * np.power(c0_norm, 4.0) * integral_val

    def get_factor(self, e):
        # 处理边界和解析延拓
        if e < 1e-4: return 1.0
        # 修正系数为 768/425
        if e > 0.99999:
            return (768.0 / 425.0) * np.power(1 - e ** 2, 3.5)
        return self.interpolator(e)
# 初始化全局加速器实例
_ACCELERATOR = MergerTimeAccelerator()
# tmerger 函数保持不变，直接调用 _ACCELERATOR.get_factor
def tmerger_integral(m1, m2, a0, e0):
    beta = 64.0 / 5.0 * m1 * m2 * (m1 + m2)
    if beta == 0: return 1e99

    t_circ = np.power(a0, 4.0) / (4.0 * beta)

    if np.isscalar(e0):
        factor = _ACCELERATOR.get_factor(e0)
    else:
        factor = np.array([_ACCELERATOR.get_factor(e) for e in e0])

    return t_circ * factor
def GWtime(m1, m2, a1, e1):
    return tmerger_integral(m1, m2, a1, e1)
def GWtime_old(m1, m2, a1, e1):
    if e1 >= 1.0 or a1 <= 0: return 0.0
    factor = 1.6e13
    return factor * (2 / m1 / m2 / (m1 + m2)) * np.power(a1 / 0.1, 4.0) * np.power(1 - e1 * e1, 7 / 2)


def peters_factor_func(e):
    if e < 1e-10: return 0.0
    term1 = np.power(e, 12.0 / 19.0)
    term2 = 1.0 - e * e
    term3 = np.power(1.0 + (121.0 / 304.0) * e * e, 870.0 / 2299.0)
    return (term1 / term2) * term3


def solve_ae_after_time(m1, m2, a0, e0, dt):
    current_life = GWtime(m1, m2, a0, e0)
    if dt >= current_life:
        return 0.0, 0.0
    t_rem_target = current_life - dt
    c0 = a0 / peters_factor_func(e0)
    try:
        e_curr = brentq(lambda e: GWtime(m1, m2, c0 * peters_factor_func(e), e) - t_rem_target,
                        1e-50, e0, xtol=1e-12, maxiter=50)
    except:
        e_curr = e0
    a_curr = c0 * peters_factor_func(e_curr)
    return a_curr, e_curr


# ==============================================================================
# Unified Noise Handling (Log-Log Interpolation & Injection Support)
# ==============================================================================

# --- Global Noise Data Storage ---
_LISA_NOISE_DATA = None


def _try_load_lisa_noise():
    """
    尝试加载 LISA_noise_ASD.csv。
    支持 Log-Log 预计算，并兼容 core.py 的注入机制。
    自动搜索当前目录和上一级目录。
    """
    global _LISA_NOISE_DATA
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 搜索路径策略：先找当前目录，再找上一级目录（兼容不同包结构）
        possible_paths = [
            os.path.join(current_dir, 'LISA_noise_ASD.csv'),
            os.path.join(os.path.dirname(current_dir), 'LISA_noise_ASD.csv')
        ]

        file_path = None
        for p in possible_paths:
            if os.path.exists(p):
                file_path = p
                break

        if file_path:
            # 尝试读取
            try:
                data = np.loadtxt(file_path, delimiter=',')
            except ValueError:
                data = np.loadtxt(file_path, delimiter=',', skiprows=1)

            # 1. 排序与清洗
            sort_idx = np.argsort(data[:, 0])
            sorted_data = data[sort_idx]

            f_data = sorted_data[:, 0]
            asd_data = sorted_data[:, 1]

            # 过滤非正值
            mask = (f_data > 0) & (asd_data > 0)
            f_data = f_data[mask]
            asd_data = asd_data[mask]

            # 2. 预计算 Log10 数据 (用于 Log-Log 插值)
            log_f = np.log10(f_data)
            log_asd = np.log10(asd_data)

            # 3. 计算低频延拓斜率 (Slope)
            # y = kx + b -> slope = (y1-y0)/(x1-x0)
            if len(log_f) >= 2:
                low_f_slope = (log_asd[1] - log_asd[0]) / (log_f[1] - log_f[0])
            else:
                low_f_slope = -2.5  # Fallback default

            _LISA_NOISE_DATA = {
                'f_min': f_data[0],
                'f_max': f_data[-1],
                'log_f': log_f,  # 存储 log(f)
                'log_asd': log_asd,  # 存储 log(ASD)
                'low_f_slope': low_f_slope,
                'log_f_0': log_f[0],
                'log_asd_0': log_asd[0],
                'use_file': True  # 标记位
            }
            # print(f"[Info] Loaded LISA noise from {os.path.basename(file_path)}")
        else:
            print(f"[Warning] LISA noise file not found. Using analytical fallback.")
            _LISA_NOISE_DATA = None

    except Exception as e:
        print(f"[Warning] Failed to load LISA noise file ({e}). Using analytical fallback.")
        _LISA_NOISE_DATA = None


# 初始化加载
_try_load_lisa_noise()


# --- SNR Functions ---

def _S_gal_N2A5_scalar(f):
    # Analytical Galactic Background (N2A5 Model)
    if f >= 1.0e-5 and f < 1.0e-3: return np.power(f, -2.3) * 10 ** -44.62 * 20.0 / 3.0
    if f >= 1.0e-3 and f < 10 ** -2.7: return np.power(f, -4.4) * 10 ** -50.92 * 20.0 / 3.0
    if f >= 10 ** -2.7 and f < 10 ** -2.4: return np.power(f, -8.8) * 10 ** -62.8 * 20.0 / 3.0
    if f >= 10 ** -2.4 and f <= 0.01: return np.power(f, -20.0) * 10 ** -89.68 * 20.0 / 3.0
    return 0.0


S_gal_N2A5 = np.vectorize(_S_gal_N2A5_scalar)


def _S_n_lisa_original(f):
    """Fallback Analytical Model (Robson+19 / N2A5)"""
    m1 = 5.0e9
    m2 = sciconsts.c * 0.41 / m1 / 2.0
    term_inst = 20.0 / 3.0 * (1 + (f / m2) ** 2) * (
                4.0 * (9.0e-30 / (2 * pi * f) ** 4 * (1 + 1.0e-4 / f)) + 2.96e-23 + 2.65e-23) / m1 ** 2
    return term_inst + S_gal_N2A5(f)


def S_n_lisa(f):
    """
    Unified Noise Calculator:
    1. Checks if file/injected data exists in _LISA_NOISE_DATA.
    2. If yes -> Log-Log Interpolation with Slope Extrapolation.
    3. If no  -> Fallback to analytical formula.
    """
    if _LISA_NOISE_DATA is not None and _LISA_NOISE_DATA.get('use_file', False):
        f_arr = np.atleast_1d(f)
        # Convert to Log space (protect against f<=0)
        log_f_in = np.log10(np.maximum(f_arr, 1e-30))

        # 1. Log-Log Interpolation
        # left=NaN (handle later), right=0.0 (ASD=1.0 for high freq)
        log_asd_out = np.interp(
            log_f_in,
            _LISA_NOISE_DATA['log_f'],
            _LISA_NOISE_DATA['log_asd'],
            left=np.nan,
            right=0.0
        )

        # 2. Handle Low Frequency Extrapolation (Slope)
        mask_low = np.isnan(log_asd_out)
        if np.any(mask_low):
            log_asd_out[mask_low] = _LISA_NOISE_DATA['log_asd_0'] + \
                                    _LISA_NOISE_DATA['low_f_slope'] * \
                                    (log_f_in[mask_low] - _LISA_NOISE_DATA['log_f_0'])

        # 3. Convert back to Linear ASD and square to get Sn(f)
        asd_out = np.power(10.0, log_asd_out)
        res = asd_out * asd_out

        if np.isscalar(f): return res[0]
        return res
    else:
        return _S_n_lisa_original(f)

def forb(M, a):
    return 1.0 / 2.0 / pi * np.sqrt(M) * np.power(a, -1.5)
def J(n, x):
    # scipy.special.jv 原生支持数组输入
    return scipy.special.jv(n, x)
def g(n, e):
    # n 可以是数组，e 是标量
    # 这里所有的计算都会自动广播 (Broadcasting)
    ne = n * e

    term1 = J(n - 2, ne) - 2 * e * J(n - 1, ne) + 2 / n * J(n, ne) + 2 * e * J(n + 1, ne) - J(n + 2, ne)
    term2 = J(n - 2, ne) - 2 * J(n, ne) + J(n + 2, ne)
    term3 = J(n, ne)

    # 注意：n**4 可能会很大，但在 float64 下通常没问题
    result = np.power(n, 4.0) / 32.0 * (
            np.power(term1, 2.0) +
            (1 - e * e) * np.power(term2, 2.0) +
            4.0 / (3.0 * n * n) * np.power(term3, 2.0)
    )
    return result
def h0(a, m1, m2, Dl):
    return np.sqrt(32 / 5) * m1 * m2 / Dl / a
@njit(fastmath=True, cache=True)
def _get_sn_val_jit(f, use_file, log_f_grid, log_asd_grid, low_f_slope, log_f_0, log_asd_0):
    """
    JIT-friendly noise calculator (Scalar Version)
    Matches logic of S_n_lisa exactly but runs inside Numba.
    """
    if not use_file:
        # Fallback Analytical
        m1_a = 5.0e9
        m2_a = CONST_c * 0.41 / m1_a / 2.0

        # S_gal part
        gal_res = 0.0
        if f >= 1.0e-5 and f < 1.0e-3:
            gal_res = np.power(f, -2.3) * 10 ** (-44.62) * 20.0 / 3.0
        elif f >= 1.0e-3 and f < 10 ** (-2.7):
            gal_res = np.power(f, -4.4) * 10 ** (-50.92) * 20.0 / 3.0
        elif f >= 10 ** (-2.7) and f < 10 ** (-2.4):
            gal_res = np.power(f, -8.8) * 10 ** (-62.8) * 20.0 / 3.0
        elif f >= 10 ** (-2.4) and f <= 0.01:
            gal_res = np.power(f, -20.0) * 10 ** (-89.68) * 20.0 / 3.0

        term1 = 20.0 / 3.0 * (1.0 + np.power(f / m2_a, 2.0))
        term2_inner = 9.0e-30 / np.power(2 * CONST_pi * f, 4.0) * (1.0 + 1.0e-4 / f)
        term2 = 4.0 * term2_inner + 2.96e-23 + 2.65e-23

        return term1 * term2 / np.power(m1_a, 2.0) + gal_res

    else:
        # Interpolation Logic
        f_safe = max(f, 1e-30)
        log_f_in = np.log10(f_safe)

        if log_f_in < log_f_grid[0]:
            log_asd = log_asd_0 + low_f_slope * (log_f_in - log_f_0)
        elif log_f_in > log_f_grid[-1]:
            log_asd = 0.0  # ASD = 1.0 -> log = 0
        else:
            log_asd = np.interp(log_f_in, log_f_grid, log_asd_grid)

        asd = 10.0 ** log_asd
        return asd * asd
@njit(fastmath=True, cache=True)
def _dSNR_high_E_kernel(f_arr, n_vals, g_vals, delta_log_f, use_file, log_f, log_asd, slope, lf0, lasd0):
    """
    修改说明：
    1. 参数 deltaf 变为 delta_log_f
    2. 积分公式变更：
       原积分: Integral[ g / (f^2 * Sn) * df ]
       变量代换: df = f * d(ln f)
       新积分: Integral[ g / (f^2 * Sn) * f * d(ln f) ] = Integral[ g / (f * Sn) * d(ln f) ]
    """
    total = 0.0
    n_len = len(f_arr)

    for i in range(n_len):
        n = n_vals[i]
        if n > 0:
            f = f_arr[i]
            sn = _get_sn_val_jit(f, use_file, log_f, log_asd, slope, lf0, lasd0)

            if sn > 0:
                g = g_vals[i]
                # 修改点：除数从 f*f 改为 f，因为 df = f * d(log_f)
                term = g / (f * sn)
                total += term

    return total * delta_log_f
@njit(fastmath=True, cache=True)
def _dSNR_low_E_kernel(n_arr, g_vals, forb_val, h0_val, use_file, log_f, log_asd, slope, lf0, lasd0):
    # 低偏心率部分保持不变
    total = 0.0
    n_len = len(n_arr)

    # Precompute constant: 8 * h0^2
    const_num = 8.0 * h0_val * h0_val

    for i in range(n_len):
        n = n_arr[i]
        fn = n * forb_val
        sn = _get_sn_val_jit(fn, use_file, log_f, log_asd, slope, lf0, lasd0)

        if sn > 0:
            g = g_vals[i]
            # Optimization: 8 * g * h0^2 / (n^2 * Sn)
            term = (const_num * g) / (n * n * sn)
            total += term

    return total
def dSNR2dt_numpy(m1, m2, a, e, Dl):
    """
    Optimized dSNR2dt using Numba Kernels.
    Modifications: High E mode now uses log-uniform frequency grid.
    """
    fmin = 1e-8
    fmax = 0.1
    fnumber = 200

    f0 = 1. / 2. / pi * np.sqrt(m1 + m2) * np.power(a, -1.5)

    # 阈值判断标准保持线性逻辑，或者根据需要调整
    threshold = (fmax - fmin) / fnumber

    # Prepare Noise Data for JIT
    use_file = False
    log_f_g = np.array([0.0])
    log_asd_g = np.array([0.0])
    slope = 0.0
    lf0 = 0.0
    lasd0 = 0.0

    if _LISA_NOISE_DATA is not None and _LISA_NOISE_DATA.get('use_file', False):
        use_file = True
        log_f_g = _LISA_NOISE_DATA['log_f']
        log_asd_g = _LISA_NOISE_DATA['log_asd']
        slope = _LISA_NOISE_DATA['low_f_slope']
        lf0 = _LISA_NOISE_DATA['log_f_0']
        lasd0 = _LISA_NOISE_DATA['log_asd_0']

    # --- High Eccentricity Mode (修改部分) ---
    if f0 * 10 < threshold:
        h = np.sqrt(32. / 5.) * m1 * m2 / Dl / a

        # 修改 1: 使用 geomspace 生成对数均匀分布点
        f_arr = np.geomspace(fmin, fmax, fnumber)

        # 修改 2: 计算对数步长 d(ln f)
        # d(ln f) = ln(fmax / fmin) / (N - 1)
        delta_log_f = np.log(fmax / fmin) / (fnumber - 1)

        # 2. Vectorized n calculation
        n_vals = (f_arr / f0).astype(np.int64)

        # 3. Calculate g using SciPy
        # 确保外部的 g 函数能处理 n_vals
        g_vals = g(n_vals, e)

        # 4. Kernel Summation (JIT)
        # 传入 delta_log_f 而不是 deltaf
        raw_sum = _dSNR_high_E_kernel(
            f_arr, n_vals, g_vals, delta_log_f,
            use_file, log_f_g, log_asd_g, slope, lf0, lasd0
        )

        return raw_sum * 8 * h * h * f0

    # --- Low Eccentricity Mode (保持不变) ---
    else:
        nmin = int(fmin / f0) + 1
        nmax = int(fmax / f0) + 2

        n_arr = np.arange(nmin, nmax)
        if len(n_arr) == 0:
            return 0.0

        # 1. Calculate g (SciPy)
        g_vals = g(n_arr, e)

        val_h0 = h0(a, m1, m2, Dl)
        val_forb = forb(m1 + m2, a)

        # 2. Kernel Summation (JIT)
        total_sum = _dSNR_low_E_kernel(
            n_arr, g_vals, val_forb, val_h0,
            use_file, log_f_g, log_asd_g, slope, lf0, lasd0
        )

        return total_sum
def SNR_analytical_geo(m1, m2, a, e, tobs, Dl):
    m1=m1*m_sun
    m2=m2*m_sun
    a=a*AU
    Dl=Dl*1e3*pc
    tobs=tobs*years
    # 1. Merger Time Check
    tmerger = tmerger_lower(m1, m2, a, e)
    used_tobs = tobs

    if tmerger <= tobs:
        tmerger = tmerger(m1,m2,a,e)
        if tmerger <= tobs:
            #print(f"[Warning] System evolves too fast! analytical lower bound of tmerger ({tmerger:.2e} s) < tobs ({tobs:.2e} s).")
            #print(f"Approximation inaccurate. Adjusting tobs to : {tmerger}")
            used_tobs = tmerger

    # 2. Calculate Rate (Using Numpy Vectorization)
    rate_val = dSNR2dt_numpy(m1, m2, a, e, Dl)

    # 3. Final SNR^2
    final_snr2 = rate_val * used_tobs

    return np.sqrt(final_snr2)
def tmerger_lower(m1, m2, a, e):
    beta = 64 / 5 * m1 * m2 * (m1 + m2)
    tc = np.power(a, 4) / (4 * beta)
    t = tc * np.power(1 - e * e, 7 / 2)
    return t


def SNR_analytical_geo0(m1_sol, m2_sol, a_au, e, tobs_yr, Dl_kpc):
    if a_au <= 0 or e >= 1.0: return 0.0
    m1_s = m1_sol * m_sun_sec
    m2_s = m2_sol * m_sun_sec
    a_s = a_au * AU_sec
    Dl_s = Dl_kpc * 1000.0 * pc_sec
    tobs_s = tobs_yr * year_sec
    rp_s = a_s * (1 - e)
    term_f = (m1_s + m2_s) / (4 * pi * pi * np.power(rp_s, 3.0))
    f0max = 2 * np.sqrt(term_f)
    h0max = np.sqrt(32 / 5) * m1_s * m2_s / (Dl_s * a_s * (1 - e))
    Sn_val = S_n_lisa(f0max)
    if Sn_val <= 0: return 0.0
    sqrtsnf = np.sqrt(Sn_val)
    snrcur = h0max / sqrtsnf * np.sqrt(tobs_s * np.power(1 - e, 3 / 2))
    return snrcur


# ==========================================
# 4. Data Management Class (Lazy Loading & Filtering)
# ==========================================

class _GNBBHInternalManager:
    def __init__(self, filename_gn="evolution_history.npy", filename_ync="evolution_history_YNC.npy"):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path_gn = os.path.join(current_script_dir, 'data', filename_gn)
        self.file_path_ync = os.path.join(current_script_dir, 'data', filename_ync)

        # Lazy Loading: Data is None initially
        self.raw_data_gn = None
        self.raw_data_ync = None

        # Filtering State
        self.current_max_mass = 100.0  # Default Threshold

        self.efinal_inv_cdf = None
        self.sorted_efinal_for_plot = None
        self.merged_indices = []

    def check_and_update_threshold(self, new_max_mass):
        """
        Public functions call this first.
        If the requested mass threshold differs from what is currently loaded,
        clear the cache to force a reload and re-filter.
        """
        if abs(self.current_max_mass - new_max_mass) > 1e-6:
            # print(f"[Manager] Threshold changed ({self.current_max_mass} -> {new_max_mass}). Reloading data...")
            self.current_max_mass = new_max_mass
            self.raw_data_gn = None
            self.raw_data_ync = None
            self.merged_indices = []
            self.efinal_inv_cdf = None

    def _ensure_gn_loaded(self):
        if self.raw_data_gn is None:
            self._load_data(self.file_path_gn, is_ync=False)
            self._build_merger_statistics()

    def _ensure_ync_loaded(self):
        if self.raw_data_ync is None:
            self._load_data(self.file_path_ync, is_ync=True)

    def _load_data(self, path, is_ync=False):
        """
        Loads data and IMMEDIATELY filters out systems where m1 or m2 > self.current_max_mass.
        """
        label = "YNC" if is_ync else "GN"
        if os.path.exists(path):
            # 1. Load raw data
            data = np.load(path, allow_pickle=True)
            original_len = len(data)

            # 2. Filter data based on current_max_mass
            # Columns: 1 is m1, 2 is m2
            if len(data) > 0:
                mask = (data[:, 1] <= self.current_max_mass) & (data[:, 2] <= self.current_max_mass)
                data = data[mask]

            # print(f"[{label}_BBH] Loaded and filtered (Max M={self.current_max_mass}): {len(data)}/{original_len} systems.")

            if is_ync:
                self.raw_data_ync = data
            else:
                self.raw_data_gn = data
        else:
            print(f"[Warning] File {path} not found.")
            if is_ync:
                self.raw_data_ync = []
            else:
                self.raw_data_gn = []

    def _build_merger_statistics(self):
        # Operates on already filtered self.raw_data_gn
        if len(self.raw_data_gn) == 0: return
        e_vals = []
        indices = []
        for i, sys_data in enumerate(self.raw_data_gn):
            a_fin = sys_data[9]
            e_fin = sys_data[10]
            if a_fin < 1e-2:
                e_vals.append(e_fin)
                indices.append(i)
        self.merged_indices = indices
        if len(e_vals) > 0:
            e_arr = np.sort(np.array(e_vals))
            y_vals = np.arange(1, len(e_arr) + 1) / len(e_arr)
            self.efinal_inv_cdf = sci_interpolate.interp1d(
                y_vals, e_arr, kind='linear', bounds_error=False, fill_value=(e_arr[0], e_arr[-1])
            )
            self.sorted_efinal_for_plot = e_arr

    def get_random_mergers(self, n):
        self._ensure_gn_loaded()
        if len(self.merged_indices) == 0: return []
        chosen_inds = random.choices(self.merged_indices, k=n)
        return [self.raw_data_gn[i] for i in chosen_inds]

    def generate_ecc_from_cdf(self, n):
        self._ensure_gn_loaded()
        if self.efinal_inv_cdf is None: return np.zeros(n)
        u = np.random.uniform(0, 1, n)
        return self.efinal_inv_cdf(u)

    def generate_snapshot_objects(self, Gamma_rep, ync_age=None, ync_count=0,Tobs_yr=10.0,dist_kpc = 8.0):
        """
        Logic remains identical to original source code.
        Data is already filtered at load time.
        """
        # Ensure data is loaded (filtered by current threshold)
        if Gamma_rep > 0: self._ensure_gn_loaded()
        if ync_count > 0: self._ensure_ync_loaded()

        mwGNsnapshot = []


        # --- Helper to process system ---
        def process_and_add(system, current_age, label):
            res = self._get_system_state_at_time(system, current_age)
            if res is not None:
                # res is (a, e, m1, m2)
                a, e, m1, m2 = res
                snr = SNR_analytical_geo(m1, m2, a, e, Tobs_yr, dist_kpc)
                mwGNsnapshot.append([label, dist_kpc, a, e, m1, m2, snr])

        # --- 1. GN Steady State ---
        if self.raw_data_gn is not None and len(self.raw_data_gn) > 0 and Gamma_rep > 0:
            lifetimes = np.array([row[8] for row in self.raw_data_gn])
            t_final_max = np.max(lifetimes)
            window_myr = t_final_max / 1e6
            total_systems_to_gen = int(window_myr * Gamma_rep)

            birth_times = np.random.uniform(-t_final_max, 0, total_systems_to_gen)
            template_indices = np.random.randint(0, len(self.raw_data_gn), total_systems_to_gen)

            for i in range(total_systems_to_gen):
                sys_idx = template_indices[i]
                t_start = birth_times[i]
                process_and_add(self.raw_data_gn[sys_idx], -t_start, 'GN_Steadystate')

        # --- 2. YNC ---
        if self.raw_data_ync is not None and len(self.raw_data_ync) > 0 and ync_count > 0 and ync_age is not None:
            ync_indices = np.random.randint(0, len(self.raw_data_ync), int(ync_count))
            for sys_idx in ync_indices:
                process_and_add(self.raw_data_ync[sys_idx], ync_age, 'GN_YNC')

        return mwGNsnapshot

    def _get_system_state_at_time(self, system, current_age):
        m1, m2, tf_actual = system[1], system[2], system[8]
        snapshots = system[11]

        if current_age > tf_actual: return None

        a_curr, e_curr = -1, -1
        if len(snapshots) > 0:
            snaps_arr = np.array(snapshots)
            times = snaps_arr[:, 0]
            if current_age <= times[-1]:
                idx = (np.abs(times - current_age)).argmin()
                a_curr, e_curr = snaps_arr[idx, 1], snaps_arr[idx, 2]
            else:
                t_last, a_last, e_last = times[-1], snaps_arr[-1, 1], snaps_arr[-1, 2]
                dt = current_age - t_last
                if dt > 0:

                    a_curr, e_curr = solve_ae_after_time(m1, m2, a_last, e_last, dt)
                else:
                    a_curr, e_curr = a_last, e_last
        else:
            a_curr, e_curr = system[3], system[4]

        if a_curr > 0: return (a_curr, e_curr, m1, m2)
        return None


_manager = _GNBBHInternalManager()


# ==========================================
# 5. Public API Functions
# ==========================================

def generate_random_merger_eccentricities(n=1000, max_bh_mass=100.0):
    """
    Returns random merged eccentricities.
    Filter: m1, m2 <= max_bh_mass (default 100).
    """
    _manager.check_and_update_threshold(max_bh_mass)
    return _manager.generate_ecc_from_cdf(n)


def plot_ecc_cdf_log(e_list=None, max_bh_mass=100.0):
    """
    Plots CDF of log(e).
    If e_list is None, loads data using max_bh_mass filter.
    """
    if e_list is None:
        _manager.check_and_update_threshold(max_bh_mass)
        _manager._ensure_gn_loaded()
        if not hasattr(_manager, 'sorted_efinal_for_plot'): return
        data = _manager.sorted_efinal_for_plot
        label = f"GN Mergers (M<{max_bh_mass})"
    else:
        data = np.array(e_list)
        label = "Sample"

    valid_mask = data > 1e-50
    if np.sum(valid_mask) == 0: return
    e_valid = data[valid_mask]
    log_e = np.log10(e_valid)
    sorted_log_e = np.sort(log_e)
    cdf = np.arange(1, len(sorted_log_e) + 1) / len(sorted_log_e)

    plt.figure(figsize=(7, 6))
    plt.step(sorted_log_e, cdf, where='post', label=f"{label} (N={len(e_valid)})", lw=2)
    plt.xlabel(r"$\log_{10}(e)$ @10Hz", fontsize=16)
    plt.ylabel("CDF", fontsize=16)
    plt.title("Eccentricity of Merging BBHs in LIGO band", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def get_random_merger_systems(n=10, max_bh_mass=100.0):
    """
    Returns N random merger systems as a list of parameters.
    Filter: m1, m2 <= max_bh_mass (default 100).
    Format: [m1, m2, ai, ei, i_i, a2, afinal, efinal, t_final]
    """
    _manager.check_and_update_threshold(max_bh_mass)
    raw_sys = _manager.get_random_mergers(n)
    result = []
    for s in raw_sys:
        # s indices: 1:m1, 2:m2, 3:ai, 4:ei, 6:ii, 5:a2, 9:afin, 10:efin, 8:t_final
        sys_list = [s[1], s[2], s[3], s[4], s[6], s[5], s[9], s[10], s[8]]
        result.append(sys_list)
    return result


def generate_snapshot_population(Gamma_rep=3.0, ync_age=None, ync_count=0, max_bh_mass=100.0):
    """
    Generates the snapshot object list.
    Filter: m1, m2 <= max_bh_mass (default 100) applied at load time.
    Returns: list of [label, dist, a, e, m1, m2, snr]
    """
    _manager.check_and_update_threshold(max_bh_mass)
    return _manager.generate_snapshot_objects(Gamma_rep, ync_age, ync_count)


def plot_snapshot_population(mwGNsnapshot, title="MW Galactic Nucleus BBH Snapshot"):
    """
    Plots the snapshot population list.
    mwGNsnapshot: list of [label, dist, a, e, m1, m2, snr]
    """
    if not mwGNsnapshot:
        print("Empty snapshot.")
        return

    # Extract data columns
    data = np.array(mwGNsnapshot, dtype=object)
    # col indices: 2:a, 3:e, 6:snr
    a_arr = data[:, 2].astype(float)
    e_arr = data[:, 3].astype(float)
    snr_arr = data[:, 6].astype(float)
    ome_arr = 1.0 - e_arr

    # Sort for plotting (High SNR on top)
    idx = np.argsort(snr_arr)
    a_p = a_arr[idx]
    ome_p = ome_arr[idx]
    snr_p = snr_arr[idx]

    my_cmap = copy.copy(mpl.colormaps['jet'])
    my_cmap.set_over('red')
    my_cmap.set_under(my_cmap(0))

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(a_p, ome_p,
                     s=np.clip(np.sqrt(snr_p) * 30, 10, 400),
                     c=snr_p,
                     cmap=my_cmap,
                     norm=mcolors.LogNorm(vmin=0.1, vmax=100),
                     alpha=1, edgecolors='k', linewidths=0.3)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Semi-major Axis [au]", fontsize=14)
    plt.ylabel(r"$1 - e$", fontsize=14)
    plt.title(f"{title}\nTotal Systems: {len(mwGNsnapshot)}", pad=15, fontsize=14)
    plt.grid(True, which="both", alpha=0.15)

    cbar = plt.colorbar(sc, extend='both', aspect=30)
    cbar.set_label(r'SNR (10yr LISA)', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.show()


def generate_and_plot_snapshot(Gamma_rep=3.0, ync_age=None, ync_count=0, max_bh_mass=100.0, title="MW Snapshot"):
    """Wrapper to generate and plot in one go."""
    snap = generate_snapshot_population(Gamma_rep, ync_age, ync_count, max_bh_mass=max_bh_mass)
    plot_snapshot_population(snap, title)