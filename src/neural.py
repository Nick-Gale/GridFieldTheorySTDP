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


def _xp(use_gpu: bool = False):
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
    xp = kernel

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


# --------------------------------------------------------------------
# Main class ----------------------------------------------------------
# --------------------------------------------------------------------
class System:
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
        params: Dict[str, float] | None = None,
        *,
        use_gpu: bool = False,
        init_activity: Iterable[float] | None = None,
    ):
        self.xp = _xp(use_gpu)
        self.shape = shape
        self.dt = dt
        self.H, self.W = shape
        self.N = self.H * self.W

        # ----- parameters -----
        self.params = params or {}
        self.params.setdefault("alpha", 1.0)
        self.params.setdefault("theta", 0.0)
        self.params.setdefault("sigma1", 5.0)
        self.params.setdefault("sigma2", 15.0)
        self.params.setdefault("A1", 1.0)
        self.params.setdefault("A2", 0.5)

        # ----- initial weights -----
        kernel = _wizard_hat_kernel(shape, self.params)      # (H, W)
        self._W_flat = _build_weight_flat(shape, kernel)    # (N, N)

        # Provide a view of the weight as 4‑D (H, W, H, W)
        self.W = self._W_flat.reshape(self.H, self.W, self.H, self.W)

        # ----- state -----
        self.activity = self.xp.zeros((self.H, self.W), dtype=self.xp.float32)
        if init_activity is not None:
            self.activity[:] = init_activity

        # ----- history -----
        self.history = self.xp.zeros((1000, self.H, self.W), dtype=self.xp.float32)
        self._history_index = 0

    # ------------------------------------------------------------------
    # Firing‑rate nonlinearity -----------------------------------------
    # ------------------------------------------------------------------
    def _firing(self, u: "xp.ndarray") -> "xp.ndarray":
        """Sigmoid firing‑rate function."""
        alpha = self.params["alpha"]
        theta = self.params["theta"]
        return 1.0 / (1.0 + self.xp.exp(-alpha * (u - theta)))

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
    def propagate(self, current: "xp.ndarray") -> None:
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
        self._history_index = (self._history_index + 1) % 1000

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
