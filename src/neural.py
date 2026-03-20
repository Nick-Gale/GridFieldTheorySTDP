# Module for the Neural and STDP solvers.

from __future__ import annotations

import numpy as np
from typing import Dict, Iterable, Tuple

# --------------------------------------------------------------------
# Optional GPU support -------------------------------------------------
# --------------------------------------------------------------------
try:
    import cupy as cp  # pragma: no cover
    _gpu_available = True
except Exception:  # pragma: no cover
    cp = None
    _gpu_available = False


def _xp(use_gpu: bool = True):
    """Return the appropriate array module (NumPy / CuPy)."""
    return cp if use_gpu and _gpu_available else np


# --------------------------------------------------------------------
# Helper functions -----------------------------------------------------
# --------------------------------------------------------------------
def _wizard_hat_kernel(shape: Tuple[int, int], params: Dict[str, float]) -> "xp.ndarray":
    """
    Return a 2‑D Wizard‑Hat kernel (difference of Gaussians) with periodic
    boundary conditions.  The kernel is centered at the origin.

    Parameters
    ----------
    shape : Tuple[int, int]
        Grid size `(H, W)`.
    params : dict
        Kernel parameters (`sigma1`, `sigma2`, `A1`, `A2`).

    Returns
    -------
    k : xp.ndarray
        Kernel of shape `(H, W)`.  The element `k[dx, dy]` corresponds to
        a shift of `dx` rows and `dy` columns.
    """
    H, W = shape
    xp = _xp()
    x = xp.arange(-W // 2, W // 2, dtype=xp.float32)
    y = xp.arange(-H // 2, H // 2, dtype=xp.float32)
    X, Y = xp.meshgrid(x, y, indexing="xy")
    r2 = X ** 2 + Y ** 2

    sigma1 = params.get("sigma1", 5.0)
    sigma2 = params.get("sigma2", 15.0)
    A1 = params.get("A1", 1.0)
    A2 = params.get("A2", 0.5)

    g1 = A1 * xp.exp(-r2 / (2 * sigma1 ** 2))
    g2 = A2 * xp.exp(-r2 / (2 * sigma2 ** 2))
    return g1 - g2  # excitatory centre, inhibitory surround


def _STDP_kernel(N: int, params: Dict[str, float]) -> "xp.ndarray":
    """
    Return an asysmetric STDP kernel
    Parameters
    ----------
    N : int
    params : dict
        Kernel parameters (`W0`, `tau`).

    Returns
    -------
    k : xp.ndarray
        Kernel of shape `(N,)`.  The element `k[i]` corresponds to
        the temporal corelation of i-shifted time units.
    """
    xp = _xp()
    T = xp.arange(-N // 2, N // 2, dtype=xp.float32)

    sigma1 = params.get("w0", 1.0)
    tau = params.get("tau", 20)
    
    k = np.zeros(N)
    for t in T:
        k[int(t)] = xp.sign(t) * xp.exp(-xp.abs(t) * tau)
    return k  


def _build_weight_flat(shape: Tuple[int, int], kernel: "xp.ndarray") -> "xp.ndarray":
    """
    Build the 4‑D weight tensor from a shift‑invariant kernel and return
    a flattened 2‑D matrix of shape `(N, N)`.

    The element `(s, t)` of the returned matrix is the weight from
    source `s` to target `t`, where
    `s = i*W + j`, `t = k*W + l`.

    Parameters
    ----------
    shape : Tuple[int, int]
        Grid size `(H, W)`.
    kernel : xp.ndarray
        2‑D kernel of shape `(H, W)` that gives the weight for a
        relative displacement `(k-i, l-j)`.

    Returns
    -------
    W_flat : xp.ndarray
        Matrix of shape `(N, N)` ready for a matrix‑multiply convolution.
    """
    H, W = shape
    N = H * W
    xp = _xp()
    # Create index arrays
    i = xp.arange(H).reshape(-1, 1, 1, 1)      # (H,1,1,1)
    j = xp.arange(W).reshape(1, -1, 1, 1)      # (1,W,1,1)
    k = xp.arange(H).reshape(1, 1, -1, 1)      # (1,1,H,1)
    l = xp.arange(W).reshape(1, 1, 1, -1)      # (1,1,1,W)

    # Relative displacements (periodic)
    dx = (k - i) % H  # shape (H,1,H,1)
    dy = (l - j) % W  # shape (1,W,1,W)

    # Broadcast the kernel to all (i,j,k,l) combinations
    # kernel[dx, dy] -> shape (H, W, H, W)
    kernel_4d = kernel[dx, dy]  # advanced indexing with broadcasting

    # Flatten to (N, N)
    W_flat = kernel_4d.reshape(N, N)
    return W_flat


# Main class ----------------------------------------------------------
# --------------------------------------------------------------------
class CorticalSystem:
    """
    2‑D Neural‑Field Theory system (Amari model) with a **dynamic** weight
    tensor.

    Parameters
    ----------
    shape : Tuple[int, int]
        Spatial discretisation `(H, W)`.
    dt : float
        Integration time step.
    params : dict, optional
        Dictionary of scalar parameters.  Accepted keys:

            * `alpha`  – slope of the firing‑rate sigmoid
            * `theta`  – firing‑rate threshold
            * `sigma1` – width of the excitatory Gaussian
            * `sigma2` – width of the inhibitory Gaussian
            * `A1`     – amplitude of the excitatory Gaussian
            * `A2`     – amplitude of the inhibitory Gaussian

    use_gpu : bool, optional
        If True and CuPy is available, the solver will run on the GPU.
    init_activity : Iterable[float], optional
        Initial activity pattern.  Must be broadcastable to `(H, W)`.
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        dt: float,
        T: int,
        params: Dict[str, float] | None = None,
        *,
        use_gpu: bool = True,
        init_activity: Iterable[float] | None = None,
    ):
        self.xp = _xp(use_gpu)
        self.shape = shape
        self.dt = dt
        self.H, self.W = shape
        self.N = self.H * self.W
        self.T = T

        # ----- parameters -----
        # activity
        self.params = params or {}
        self.params.setdefault("alpha", 1.0)
        self.params.setdefault("theta", 0.0)
        self.params.setdefault("sigma1", 5.0)
        self.params.setdefault("sigma2", 15.0)
        self.params.setdefault("T", 100)
        
        # kernel
        self.params.setdefault("A1", 1.0)
        self.params.setdefault("A2", 0.5)
        
        # STDP
        self.params.setdefault("tau", 20)
        self.params.setdefault("w0", 1)

        # ----- initial weights -----
        kernel = _wizard_hat_kernel(shape, self.params)      # (H, W)
        self._W_flat = _build_weight_flat(shape, kernel)    # (N, N)
        self.STDP = _STDP_kernel(self.N, self.params)

        # Provide a view of the weight as 4‑D (H, W, H, W)
        self.weights = self._W_flat.reshape(self.H, self.W, self.H, self.W)

        # ----- state -----
        self.activity =self.xp.zeros((self.H, self.W), dtype=self.xp.float32)
        if init_activity is not None:
            self.activity[:] = init_activity

        # ----- history -----
        self.history = self.xp.zeros((self.T, self.H, self.W), dtype=self.xp.float32)
        self._history_index = 0
    
    # ------------------------------------------------------------------
    # STDP Kernel Update -----------------------------------------------
    # ------------------------------------------------------------------
    def _weighted_correlation(self, x, y, kernel) -> float:
        """
        Compute a weighted correlation between two equal‑length 1‑D arrays.
    
        Parameters
        ----------
        x, y   : 1-D numpy arrays of the same length
        kernel : 1-D array of non‑negative weights, length = len(x)
                 It is assumed that kernel[0] corresponds to lag 0,
                 kernel[1] to lag +1, kernel[-1] to lag -(len(x)-1).
    
        Returns
        -------
        float : weighted correlation coefficient
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
    
        N = len(x)
        k = np.array(kernel, dtype=float)
    
        mx = x.mean()
        my = y.mean()
    
        # Weighted sums
        num = 0.0; den_x = 0.0; den_y = 0.0
    
        for tau in range(-N+1, N):              # all possible lags
            w = k[abs(tau)] if abs(tau) < N else 0.0
            if tau < 0:                         # X leads Y
                ix = self.xp.arange(-tau, N)         # indices of X that align with Y
                iy = self.xp.arange(0, N+tau)
            else:                               # Y leads X
                ix = self.xp.arange(0, N-tau)
                iy = self.xp.arange(tau, N)
            # Compute contribution at this lag
            num += w * self.xp.sum((x[ix] - mx) * (y[iy] - my))
            den_x += w * self.xp.sum((x[ix] - mx) ** 2)
            den_y += w * self.xp.sum((y[iy] - my) ** 2)
    
        denom = self.xp.sqrt(den_x * den_y)
        return num / denom if denom != 0 else 0.0
    
    # ------------------------------------------------------------------
    # Firing‑rate nonlinearity -----------------------------------------
    # ------------------------------------------------------------------
    def _firing(self, u: "xp.ndarray") -> "xp.ndarray":
        """Sigmoid firing‑rate function."""
        alpha = self.params["alpha"]
        theta = self.params["theta"]
        return u * self.xp.sign(u)  # 1.0 / (1.0 + self.xp.exp(-alpha * (u - theta)))

    # ------------------------------------------------------------------
    # Right‑hand side of the Amari differential equation ----------------
    # ------------------------------------------------------------------
    def _rhs(self, u: "xp.ndarray", current: "xp.ndarray") -> "xp.ndarray":
        """du/dt = -u + W * f(u) + current"""
        f_u = self._firing(u)                    # (H, W)
        f_u_flat = f_u.reshape(self.N)           # (N,)
        conv_flat = self.xp.dot(self._W_flat.T, f_u_flat)  # (N,)
        conv = conv_flat.reshape(self.H, self.W)  # (H, W)
        return -u + conv + current

    # ------------------------------------------------------------------
    # Propagation step – RK4 --------------------------------------------
    # ------------------------------------------------------------------
    def propagate(self, current: "xp.ndarray", T) -> None:
        """
        Advance the system by one time step using RK4 integration.

        Parameters
        ----------
        current : xp.ndarray
            External input for this time step, shape must match
            `self.activity` (i.e. `(H, W)`).
        """
        u = self.activity
        dt = self.dt

        k1 = dt * self._rhs(u, current)
        k2 = dt * self._rhs(u + 0.5 * k1, current)
        k3 = dt * self._rhs(u + 0.5 * k2, current)
        k4 = dt * self._rhs(u + k3, current)

        self.activity = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        # ---- history (circular buffer) ----
        self.history[self._history_index] = self.activity
        self._history_index = (self._history_index + 1) % T

    # ------------------------------------------------------------------
    # Learning step – STDP ---------------------------------------------
    # ------------------------------------------------------------------
    def learn(self) -> None:
        """
        Take 1000ms of simulated current and apply an STDP learning rule.
        """
        u_flat = self.history.reshape(self.T, self.N) 
        for i in self.xp.arange(self.N):
            for j in self.xp.arange(self.N):
                dW = self._weighted_correlation(self._firing(u_flat[:,i]), self._firing(u_flat[:,j]), self.STDP)        
                self._W_flat += dW
                # self.weights = self._W_flat.reshape(self.H, self.W, self.H, self.W)

    # ------------------------------------------------------------------
    # History access ----------------------------------------------------
    # ------------------------------------------------------------------
    def get_history(self) -> "xp.ndarray":
        """Return history in chronological order (oldest first)."""
        if self._history_index == 0:
            return self.history
        return self.xp.vstack(
            (
                self.history[self._history_index:],
                self.history[:self._history_index],
            )
        )

    def history_as_numpy(self) -> np.ndarray:
        """Return the history as a NumPy array (CPU)."""
        return np.array(self.get_history())
