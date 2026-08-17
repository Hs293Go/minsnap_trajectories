"""
Copyright © 2023 Hs293Go

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import scipy.linalg as la
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

Algorithm = Literal["closed-form", "constrained"]

_DERIVATIVE_ATTRS = ("position", "velocity", "acceleration", "jerk", "snap")


@dataclasses.dataclass(frozen=True, slots=True)
class Waypoint:
    """A waypoint defining a piecewise-polynomial trajectory.

    Holds a timestamp, a position, and optionally any of the higher-order
    derivatives (velocity, acceleration, jerk, snap). Unspecified derivatives
    are left free for the planner to choose, except that velocity and
    acceleration are pinned to zero at the first and last waypoint when
    not given explicitly.

    Parameters
    ----------
    time : float
        Timestamp of the waypoint.
    position : ArrayLike
        Position at the waypoint.
    velocity, acceleration, jerk, snap : ArrayLike, optional
        Higher-order derivatives at the waypoint. Each must broadcast to the
        same shape as ``position``.
    """

    time: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64] | None = None
    acceleration: NDArray[np.float64] | None = None
    jerk: NDArray[np.float64] | None = None
    snap: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        for attr in _DERIVATIVE_ATTRS:
            value = getattr(self, attr)
            if value is None:
                continue
            object.__setattr__(self, attr, np.asarray(value, dtype=np.float64))

    def derivative(self, order: int) -> NDArray[np.float64] | None:
        """Return the derivative of the given order, or ``None`` if not set."""
        if not 0 <= order < len(_DERIVATIVE_ATTRS):
            return None
        return getattr(self, _DERIVATIVE_ATTRS[order])


class PolynomialSize(NamedTuple):
    n_poly: int  # Number of pieces
    n_cfs: int  # degree + 1
    dim: int


class PiecewisePolynomialTrajectory(NamedTuple):
    """Piecewise-polynomial trajectory.

    ``coefficients[i, j, d]`` is the j-th coefficient of the i-th piece's
    polynomial along dimension ``d``. Each piece is parameterised by absolute
    time within the piece (i.e. ``t - time_reference[i]``).
    """

    time_reference: NDArray[np.float64]
    durations: NDArray[np.float64]
    coefficients: NDArray[np.float64]


class RotorDragParameters(NamedTuple):
    """RDR rotor drag model parameters.

    ``cp`` is a velocity-dependent rotor-drag time constant. ``dh`` and ``dv``
    are the diagonal entries of ``D = diag([dh, dv, dh])``.
    """

    cp: float
    dh: float
    dv: float


class QuadrotorTrajectory(NamedTuple):
    """Quadrotor state and input trajectory.

    ``state`` is a horizontal stack ``[position, attitude, velocity]`` where
    attitude is a unit quaternion. ``input`` is ``[thrust, body_rates]``.
    """

    state: NDArray[np.float64]
    input: NDArray[np.float64]

    @property
    def position(self) -> NDArray[np.float64]:
        return self.state[:, 0:3]

    @property
    def attitude(self) -> NDArray[np.float64]:
        return self.state[:, 3:7]

    @property
    def velocity(self) -> NDArray[np.float64]:
        return self.state[:, 7:10]

    @property
    def thrust(self) -> NDArray[np.float64]:
        return self.input[:, 0]

    @property
    def body_rates(self) -> NDArray[np.float64]:
        return self.input[:, 1:4]


def compute_trajectory_derivatives(
    polys: PiecewisePolynomialTrajectory,
    t_sample: ArrayLike,
    num_orders: int,
) -> NDArray[np.float64]:
    """Sample a piecewise-polynomial trajectory.

    Parameters
    ----------
    polys : PiecewisePolynomialTrajectory
        A trajectory produced by :func:`generate_trajectory`.
    t_sample : ArrayLike
        Sample times.
    num_orders : int
        Number of derivative orders to evaluate. ``num_orders=3`` returns
        position, velocity, and acceleration; ``num_orders=4`` adds jerk.

    Returns
    -------
    NDArray
        Array of shape ``(num_orders, len(t_sample), dim)`` stacking the
        derivative samples along the first axis.
    """
    t_sample = np.atleast_1d(np.asarray(t_sample, dtype=np.float64)).ravel()
    t_ref, _, coeffs = polys
    n_pieces, n_cfs, dim = coeffs.shape

    if num_orders < 1:
        raise ValueError("num_orders must be at least 1")
    if num_orders > n_cfs:
        raise ValueError(
            f"Cannot compute {num_orders} derivative orders from a degree-{n_cfs - 1} polynomial"
        )

    len_traj = t_sample.size
    result = np.zeros((num_orders, len_traj, dim), dtype=np.float64)
    if n_pieces == 0:
        return result

    if (t_sample < t_ref[0]).any() or (t_sample > t_ref[-1]).any():
        warnings.warn("Query points outside of bounds; clamping", stacklevel=2)
        t_sample = np.clip(t_sample, t_ref[0], t_ref[-1])

    piece_idx = np.clip(np.searchsorted(t_ref, t_sample, side="right") - 1, 0, n_pieces - 1)
    tau = t_sample - t_ref[piece_idx]
    piece_coeffs = coeffs[piece_idx]  # (len_traj, n_cfs, dim)

    n = np.arange(n_cfs)
    for r in range(num_orders):
        if r == 0:
            prefactor = np.ones(n_cfs, dtype=np.float64)
        else:
            m = np.arange(r)[:, None]
            prefactor = np.prod(n[None, :] - m, axis=0).astype(np.float64)
            # prefactor is 0 for n < r since (n - n) = 0 appears in the product.
        exponents = np.maximum(n - r, 0)
        powers = tau[:, None] ** exponents[None, :]
        weighted = prefactor[None, :] * powers  # (len_traj, n_cfs)
        result[r] = (weighted[..., None] * piece_coeffs).sum(axis=1)
    return result


def flat_output_to_quadrotor_trajectory(
    trajectory_derivatives: ArrayLike,
    vehicle_mass: float,
    yaw: ArrayLike,
    yaw_rate: ArrayLike,
    drag_params: RotorDragParameters | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map flat-output samples to quadrotor attitude and inputs.

    Parameters
    ----------
    trajectory_derivatives : ArrayLike
        Array of shape ``(num_orders, n_samples, 3)`` produced by
        :func:`compute_trajectory_derivatives`. Must include up to jerk
        (``num_orders >= 4``).
    vehicle_mass : float
        Mass of the quadrotor.
    yaw, yaw_rate : ArrayLike
        Yaw angle and yaw rate samples, length ``n_samples``.
    drag_params : RotorDragParameters, optional
        RDR rotor-drag model parameters.

    Returns
    -------
    tuple[NDArray, NDArray]
        ``(attitude, inputs)`` where attitude is an ``(n_samples, 4)`` array of
        unit quaternions and inputs is ``(n_samples, 4)`` containing thrust
        and body rates.
    """
    grav_vector = np.array([0.0, 0.0, 9.81])
    trajectory_derivatives = np.atleast_3d(np.asarray(trajectory_derivatives, dtype=np.float64))

    n_ders, len_traj, n_dims = trajectory_derivatives.shape

    if n_dims != 3:
        raise ValueError(f"Incorrect dimensions for quadrotor trajectory: {n_dims} != 3")
    if n_ders < 4:
        raise ValueError(f"Need at least 4 derivative orders (jerk); got {n_ders}")

    vel = trajectory_derivatives[1, ...]
    acc = trajectory_derivatives[2, ...]
    jer = trajectory_derivatives[3, ...]

    inputs = np.empty((len_traj, 4), dtype=np.float64)
    if drag_params is not None:
        cp_term = np.linalg.norm(vel, axis=1, keepdims=True)
        w_term = 1.0 + drag_params.cp * cp_term
        with np.errstate(divide="ignore", invalid="ignore"):
            v_dot_a = np.sum(vel * acc, axis=1, keepdims=True)
            dw_term = np.where(cp_term > 0, drag_params.cp * v_dot_a / cp_term, 0.0)

        dw = w_term * acc + dw_term * vel
        w = w_term * vel
        dh_over_m = drag_params.dh / vehicle_mass

        z = acc + dh_over_m * w + grav_vector
        z_nrm = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / z_nrm

        dz = -np.cross(z, np.cross(z, jer + dh_over_m * dw, axis=1), axis=1) / z_nrm
        inputs[:, 0] = np.sum(
            z * (vehicle_mass * (acc + grav_vector) + drag_params.dv * w),
            axis=1,
        )
    else:
        z = acc + grav_vector
        z_nrm = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / z_nrm

        dz = -np.cross(z, np.cross(z, jer, axis=1), axis=1) / z_nrm
        inputs[:, 0] = np.sum(z * (vehicle_mass * (acc + grav_vector)), axis=1)

    if np.any(z[:, 2] <= -1.0 + 1e-9):
        raise ValueError(
            "Flatness map is singular: trajectory requires the body z-axis to point "
            "directly downward at some sample (cos(tilt) <= -1). Check the trajectory "
            "or sample more conservatively."
        )

    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (len_traj,))
    yaw_rate = np.broadcast_to(np.asarray(yaw_rate, dtype=np.float64), (len_traj,))

    tilt_den = np.sqrt(2.0 * (1.0 + z[:, 2]))
    tilt0 = 0.5 * tilt_den
    tilt1 = -z[:, 1] / tilt_den
    tilt2 = z[:, 0] / tilt_den
    c_half_psi = np.cos(0.5 * yaw)
    s_half_psi = np.sin(0.5 * yaw)
    attitude = np.column_stack(
        [
            tilt1 * c_half_psi + tilt2 * s_half_psi,
            tilt2 * c_half_psi - tilt1 * s_half_psi,
            tilt0 * s_half_psi,
            tilt0 * c_half_psi,
        ]
    )
    c_psi = np.cos(yaw)
    s_psi = np.sin(yaw)
    omg_den = z[:, 2] + 1.0
    omg_term = dz[:, 2] / omg_den
    inputs[:, 1:4] = np.column_stack(
        [
            (dz[:, 0] * s_psi - dz[:, 1] * c_psi - (z[:, 0] * s_psi - z[:, 1] * c_psi) * omg_term),
            (dz[:, 0] * c_psi + dz[:, 1] * s_psi - (z[:, 0] * c_psi + z[:, 1] * s_psi) * omg_term),
            (z[:, 1] * dz[:, 0] - z[:, 0] * dz[:, 1]) / omg_den + yaw_rate,
        ]
    )

    return attitude, inputs


def compute_quadrotor_trajectory(
    polys: PiecewisePolynomialTrajectory,
    t_sample: ArrayLike,
    vehicle_mass: float,
    yaw: ArrayLike | Literal["velocity"] | None = None,
    yaw_rate: ArrayLike | None = None,
    drag_params: RotorDragParameters | None = None,
) -> QuadrotorTrajectory:
    """Compute a full quadrotor state/input trajectory.

    Parameters
    ----------
    polys : PiecewisePolynomialTrajectory
        A 3D trajectory produced by :func:`generate_trajectory`.
    t_sample : ArrayLike
        Sample times.
    vehicle_mass : float
        Mass of the quadrotor.
    yaw : ArrayLike or "velocity", optional
        Yaw references. ``None`` (default) sets yaw to zero. ``"velocity"``
        aligns yaw with the velocity vector at each sample.
    yaw_rate : ArrayLike, optional
        Yaw-rate references. ``None`` (default) sets yaw rate to zero.
    drag_params : RotorDragParameters, optional
        Rotor-drag parameters.
    """
    trajectory_derivatives = compute_trajectory_derivatives(polys, t_sample, 4)
    len_traj = trajectory_derivatives.shape[1]

    if isinstance(yaw, str):
        if yaw != "velocity":
            raise ValueError(f"Unrecognized yaw mode: {yaw!r}")
        yaw_arr = np.arctan2(trajectory_derivatives[1, :, 1], trajectory_derivatives[1, :, 0])
    elif yaw is None:
        yaw_arr = np.zeros(len_traj)
    else:
        yaw_arr = np.broadcast_to(np.asarray(yaw, dtype=np.float64), (len_traj,))

    if yaw_rate is None:
        yaw_rate_arr = np.zeros(len_traj)
    else:
        yaw_rate_arr = np.broadcast_to(np.asarray(yaw_rate, dtype=np.float64), (len_traj,))

    attitudes, inputs = flat_output_to_quadrotor_trajectory(
        trajectory_derivatives, vehicle_mass, yaw_arr, yaw_rate_arr, drag_params
    )
    positions = trajectory_derivatives[0]
    velocities = trajectory_derivatives[1]

    return QuadrotorTrajectory(np.hstack([positions, attitudes, velocities]), inputs)


def generate_trajectory(
    references: Sequence[Waypoint],
    degree: int,
    *,
    idx_minimized_orders: int | Sequence[int] = 4,
    num_continuous_orders: int = 3,
    algorithm: Algorithm = "closed-form",
    optimize_options: dict[str, Any] | None = None,
) -> PiecewisePolynomialTrajectory:
    """Plan a piecewise-polynomial trajectory through a sequence of waypoints.

    Parameters
    ----------
    references : Sequence[Waypoint]
        Waypoints defining the trajectory.
    degree : int
        Polynomial degree of each piece. Must be at least 2.
    idx_minimized_orders : int or Sequence[int], optional
        Index/indices of derivative orders to minimize. Each must satisfy
        ``2 <= idx < degree``. Default is 4 (minimum snap).
    num_continuous_orders : int, optional
        Number of derivative orders constrained to be continuous across
        piece boundaries. Default is 3 (position, velocity, acceleration).
    algorithm : "closed-form" or "constrained", optional
        ``"closed-form"`` uses Bry & Roy's unconstrained reformulation;
        ``"constrained"`` uses Mellinger & Kumar's direct QP via SLSQP.
    optimize_options : dict, optional
        Options forwarded to ``scipy.optimize.minimize``. Ignored by the
        closed-form solver.
    """
    if degree < 2:
        raise ValueError("Polynomial degree too low")

    derivative_weights = np.zeros(degree)

    idx_minimized_orders = np.asarray(idx_minimized_orders, dtype=np.int32).ravel()
    if (idx_minimized_orders < 2).any():
        raise ValueError("Minimizing 0th- or 1st-order derivatives does not make sense")
    if (idx_minimized_orders >= degree).any():
        raise ValueError(
            f"Cannot minimize derivative orders >= polynomial degree ({degree}); "
            f"got {idx_minimized_orders.tolist()}"
        )
    derivative_weights[idx_minimized_orders] = 1

    if num_continuous_orders < 3:
        raise ValueError(
            f"Constraining only {num_continuous_orders} (< 3) derivatives of position "
            "(position, velocity, acceleration) usually does not make sense"
        )
    if num_continuous_orders > degree:
        raise ValueError(
            f"Cannot constrain {num_continuous_orders}-order derivatives when "
            f"polynomial is degree {degree} only"
        )
    t_ref, refs = _parse_references(references, num_continuous_orders)

    if (t_ref < 0.0).any():
        raise ValueError("Waypoint timestamp is negative")
    durations = np.diff(t_ref)
    if (durations <= 1e-8).any():
        raise ValueError("The time duration for transiting between waypoints is too small")

    poly_dim = PolynomialSize(n_poly=refs.shape[0] - 1, n_cfs=degree + 1, dim=refs.shape[2])

    solvers = {"closed-form": _solve_closed_form, "constrained": _solve_constrained}
    if algorithm not in solvers:
        raise ValueError(f"Unrecognized algorithm: {algorithm!r}")
    polys = solvers[algorithm](
        refs,
        durations,
        poly_dim,
        derivative_weights,
        num_continuous_orders,
        optimize_options,
    )
    return PiecewisePolynomialTrajectory(t_ref, durations, polys)


def _parse_references(
    references: Sequence[Waypoint], r_cts: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    len_traj = len(references)

    t_ref: list[float] = []
    trajectory_ref: list[NDArray[np.float64]] = []
    max_specified_order = len(_DERIVATIVE_ATTRS) - 1
    for idx, wp in enumerate(references):
        t_ref.append(wp.time)
        dim = len(wp.position)
        ref: list[NDArray[np.float64]] = [wp.position]

        # Warn if the user specified derivatives that exceed the continuity request.
        for order in range(r_cts, max_specified_order + 1):
            if wp.derivative(order) is not None:
                warnings.warn(
                    f"Waypoint specifies order-{order} derivative but "
                    f"num_continuous_orders={r_cts} drops it",
                    stacklevel=3,
                )
                break

        # Velocity/acceleration default to zero at endpoints, NaN (free) elsewhere.
        # Higher-order derivatives default to NaN unless explicitly given.
        for r in range(1, r_cts):
            specified = wp.derivative(r) if r <= max_specified_order else None
            if specified is not None:
                ref.append(specified)
            elif r < 3 and idx in (0, len_traj - 1):
                ref.append(np.zeros(dim))
            else:
                ref.append(np.full((dim,), np.nan))

        trajectory_ref.append(np.array(ref))

    return (
        np.asarray(t_ref, dtype=np.float64),
        np.asarray(trajectory_ref, dtype=np.float64),
    )


def _scale_coefficients(
    coeffs: NDArray[np.float64], durations: NDArray[np.float64], n_cfs: int
) -> NDArray[np.float64]:
    """Convert per-piece coefficients from normalized to absolute time.

    ``coeffs`` and the result are shape ``(n_poly, n_cfs)``; ``durations`` is
    shape ``(n_poly,)``.
    """
    scale = (1.0 / durations[:, None]) ** np.arange(0, n_cfs)[None, :]
    return scale * coeffs


def _solve_closed_form(
    refs: NDArray[np.float64],
    durations: NDArray[np.float64],
    poly_dim: PolynomialSize,
    derivative_weights: NDArray[np.float64],
    r_cts: int,
    optimize_options: dict[str, Any] | None,
) -> NDArray[np.float64]:
    if r_cts < 3:
        raise ValueError("Trajectory must be continuous up to the 2nd order (acceleration)")

    # The unconstrained reformulation inverts the map from a piece's coefficients to
    # its 2 * r_cts endpoint derivatives. With fewer coefficients than endpoint
    # derivatives that map is overdetermined, and the least-squares inverse below
    # silently yields a trajectory that does not pass through its waypoints.
    if poly_dim.n_cfs < 2 * r_cts:
        raise ValueError(
            f"The closed-form solver needs at least {2 * r_cts} coefficients per piece to "
            f"reproduce {r_cts} derivative orders at both ends of a piece, but a degree-"
            f"{poly_dim.n_cfs - 1} polynomial has only {poly_dim.n_cfs}. Raise degree to at "
            f"least {2 * r_cts - 1}, lower num_continuous_orders, or use "
            f'algorithm="constrained"'
        )

    if optimize_options is not None:
        warnings.warn(
            "Solving the trajectory generation problem in closed form; "
            "optimize_options will be ignored",
            stacklevel=2,
        )
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Q = np.zeros((n_vars, n_vars))
    for r, c_r in enumerate(derivative_weights):
        Q += c_r * la.block_diag(*_compute_Q(poly_dim.n_cfs, r, durations))

    poly_coeffs = np.zeros(poly_dim)
    A = np.zeros((r_cts * 2 * poly_dim.n_poly, poly_dim.n_cfs * poly_dim.n_poly))
    for i in range(poly_dim.n_poly):
        s = np.s_[poly_dim.n_cfs * i : poly_dim.n_cfs * (i + 1)]
        for r in range(r_cts):
            A[r_cts * 2 * i + r, s] = _compute_tvec(poly_dim.n_cfs, r, 0) / durations[i] ** r
            A[r_cts * (2 * i + 1) + r, s] = _compute_tvec(poly_dim.n_cfs, r, 1) / durations[i] ** r

    M = np.zeros((poly_dim.n_poly * 2 * r_cts, r_cts * (poly_dim.n_poly + 1)))
    for i in range(poly_dim.n_poly):
        s1 = np.s_[2 * r_cts * i : 2 * r_cts * (i + 1)]
        s2 = np.s_[r_cts * i : r_cts * (i + 2)]
        M[s1, s2] = np.eye(2 * r_cts)

    num_d = r_cts * (poly_dim.n_poly + 1)

    C = np.eye(num_d)
    fix_idx = np.flatnonzero(np.all(~np.isnan(refs), axis=2).ravel())
    free_idx = np.setdiff1d(np.arange(num_d), fix_idx)
    C = np.hstack([C[:, fix_idx], C[:, free_idx]])

    AiMC = la.lstsq(A, M @ C)[0]
    R = AiMC.T @ Q @ AiMC

    n_fix = fix_idx.size
    Rpp = R[n_fix:, n_fix:]
    Rfp = R[:n_fix, n_fix:]

    for d in range(poly_dim.dim):
        ref = refs[..., d]
        df = ref.ravel()[fix_idx]
        rhs = -(Rfp.T @ df)
        try:
            dp = la.solve(Rpp, rhs, assume_a="pos")
        except (la.LinAlgError, ValueError):
            warnings.warn(
                "Closed-form Hessian is singular; falling back to least-squares solve",
                stacklevel=2,
            )
            dp = la.lstsq(Rpp, rhs)[0]

        coeffs = np.reshape(AiMC @ np.concatenate([df, dp]), poly_dim[0:2])
        poly_coeffs[:, :, d] = _scale_coefficients(coeffs, durations, poly_dim.n_cfs)

    return poly_coeffs


def _solve_constrained(
    refs: NDArray[np.float64],
    durations: NDArray[np.float64],
    poly_dim: PolynomialSize,
    derivative_weights: NDArray[np.float64],
    r_cts: int,
    optimize_options: dict[str, Any] | None,
) -> NDArray[np.float64]:
    opts: dict[str, Any] = {"method": "SLSQP", "tol": 1e-10}
    if optimize_options is not None:
        opts.update(optimize_options)
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Q = np.zeros((n_vars, n_vars))
    for r, c_r in enumerate(derivative_weights):
        Q += c_r * la.block_diag(*_compute_Q(poly_dim.n_cfs, r, durations))

    poly_coeffs = np.zeros(poly_dim)
    Aeq_1, beq_1 = _compute_continuity_constraints(poly_dim, durations, r_cts)
    for d in range(poly_dim.dim):
        Aeq_0, beq_0 = _compute_dynamical_constraints(poly_dim, refs[:, :, d], durations)

        Aeq = np.vstack([Aeq_0, Aeq_1])
        beq = np.concatenate([beq_0, beq_1])

        constr = optimize.LinearConstraint(Aeq, beq, beq)
        soln = optimize.minimize(
            lambda x: (x @ Q @ x) / 2,
            np.zeros(n_vars),
            constraints=constr,
            jac=lambda x: Q @ x,
            **opts,
        )
        if not soln.success:
            raise RuntimeError(f"Constrained QP solver failed for dimension {d}: {soln.message}")
        coeffs = soln.x.reshape(poly_dim[0:2])
        poly_coeffs[:, :, d] = _scale_coefficients(coeffs, durations, poly_dim.n_cfs)
    return poly_coeffs


def _compute_continuity_constraints(
    poly_dim: PolynomialSize, durations: NDArray[np.float64], r_cts: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Aeq = np.zeros(((poly_dim.n_poly - 1) * r_cts, n_vars))
    beq = np.zeros((poly_dim.n_poly - 1) * r_cts)
    for i in range(poly_dim.n_poly - 1):
        s = np.s_[poly_dim.n_cfs * i : poly_dim.n_cfs * (i + 2)]
        for r in range(r_cts):
            tvec_l = _compute_tvec(poly_dim.n_cfs, r, 1) / durations[i] ** r
            tvec_r = _compute_tvec(poly_dim.n_cfs, r, 0) / durations[i + 1] ** r
            Aeq[r_cts * i + r, s] = np.concatenate([tvec_l, -tvec_r])

    return Aeq, beq


def _compute_dynamical_constraints(
    poly_dim: PolynomialSize,
    refs: NDArray[np.float64],
    durations: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    n_constrain_orders = np.count_nonzero(~np.isnan(refs), axis=1)
    Aeq = np.zeros((n_constrain_orders.sum(), n_vars))
    beq = np.zeros(n_constrain_orders.sum())

    row_its = np.concatenate([[0], n_constrain_orders.cumsum()])
    for i in range(poly_dim.n_poly + 1):
        idx, tau = (i - 1, 1.0) if i == poly_dim.n_poly else (i, 0.0)
        s = np.s_[poly_dim.n_cfs * idx : poly_dim.n_cfs * (1 + idx)]
        for r in range(n_constrain_orders[i]):
            Aeq[row_its[i] + r, s] = _compute_tvec(poly_dim.n_cfs, r, tau) / durations[idx] ** r
            beq[row_its[i] + r] = refs[i, r]
    return Aeq, beq


def _compute_Q(n_cfs: int, r: int, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    Q = np.zeros((len(tau), n_cfs, n_cfs))

    i, l = np.meshgrid(*[np.arange(r, n_cfs)] * 2, sparse=True)
    m_seq = np.arange(0, r)[:, None, None]
    k = -2 * r + 1
    Q[:, i, l] = np.prod((i - m_seq) * (l - m_seq), axis=0) / (k + i + l) * tau[:, None, None] ** k
    return Q


def _compute_tvec(n_cfs: int, r: int, tau: float) -> NDArray[np.float64]:
    tvec = np.zeros(n_cfs)
    n_seq = np.arange(r, n_cfs)
    r_seq = np.arange(0, r)[:, None]
    tvec[n_seq] = np.prod(n_seq - r_seq, axis=0) * tau ** (n_seq - r)
    return tvec
