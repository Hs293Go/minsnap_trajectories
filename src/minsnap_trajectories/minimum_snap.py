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

import warnings
from typing import NamedTuple, Optional

import numpy as np
import scipy.linalg as la
from scipy import optimize


class Reference(NamedTuple):
    time: float
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None


class PolynomialSize(NamedTuple):

    n_poly: int  # Number of references - 1
    n_cfs: int  # degree + 1
    dim: int


class PiecewisePolynomialTrajectory(NamedTuple):
    time_reference: np.ndarray
    durations: np.ndarray
    coefficients: np.ndarray


class RotorDragParameters(NamedTuple):
    cp: float
    dh: float
    dv: float


class RobotTrajectory(NamedTuple):
    state: np.ndarray
    input: np.ndarray


class QuadrotorTrajectory(RobotTrajectory):

    @property
    def position(self):
        return self.state[:, 0:3]

    @property
    def attitude(self):
        return self.state[:, 3:7]

    @property
    def velocity(self):
        return self.state[:, 7:10]

    @property
    def thrust(self):
        return self.input[:, 0]

    @property
    def body_rates(self):
        return self.input[:, 1:4]


def compute_trajectory_derivatives(polys, t_sample, order):
    t_ref, _, coeffs = polys

    _, n_cfs, dim = coeffs.shape
    if order > n_cfs - 1:
        raise ValueError(f"Kinematic order {order} > polynomial degree {n_cfs - 1}")
    len_traj = t_sample.size
    trajectory_derivatives = np.zeros((order, len_traj, dim), dtype=np.float64)

    def find_piece(t):
        if not t_ref[0] <= t <= t_ref[-1]:
            warnings.warn("Query point is outside of bounds. Clamping")
            t = np.clip(t, t_ref[0], t_ref[-1])
        idx = np.flatnonzero(t >= t_ref[:-1])[-1]
        tau = t - t_ref[idx]
        return coeffs[idx, ...], tau

    for r in range(order):
        for k, t in enumerate(t_sample):
            piece, segment_time = find_piece(t)

            trajectory_derivatives[r, k, :] = _nd_polyvals(piece, segment_time, r)
    return trajectory_derivatives


def flat_output_to_quadrotor_trajectory(
    trajectory_derivatives,
    vehicle_mass,
    yaw,
    yaw_rate,
    drag_params=None,
):
    grav_vector = np.array([0.0, 0.0, 9.81])
    trajectory_derivatives = np.asarray(trajectory_derivatives, dtype=np.float64)
    trajectory_derivatives = np.atleast_3d(trajectory_derivatives)

    n_ders, len_traj, n_dims = trajectory_derivatives.shape

    if n_dims != 3:
        raise ValueError(
            f"Incorrect dimensions for quadrotor trajectory: {n_dims} != 3"
        )

    # This should be guaranteed if this function is forwarded to from
    # compute_quadrotor_trajectory
    assert n_ders in (3, 4)

    vel = np.atleast_2d(trajectory_derivatives[1, ...])
    acc = np.atleast_2d(trajectory_derivatives[2, ...])
    jer = np.atleast_2d(trajectory_derivatives[3, ...])

    attitude = np.empty((len_traj, 4), dtype=np.float64)
    inputs = np.empty((len_traj, 4), dtype=np.float64)
    if drag_params is not None:
        cp_term = np.sqrt(np.sum(vel * vel, axis=1))
        w_term = 1.0 + drag_params.cp * cp_term
        v_dot_a = np.sum(vel * acc, axis=1)
        dw_term = drag_params.cp * v_dot_a / cp_term

        dw = w_term * acc + dw_term * vel
        w = w_term * vel
        dh_over_m = drag_params.dh / vehicle_mass

        z = acc + dh_over_m * w + grav_vector
        z_nrm = np.linalg.norm(z, axis=1, keepdims=True)
        z /= z_nrm

        dz = -np.cross(z, np.cross(z, jer + dh_over_m * dw, axis=0), axis=0) / z_nrm
        inputs[0, :] = np.sum(
            z * (vehicle_mass * (acc + grav_vector) + drag_params.dv * w), axis=0
        )
    else:
        z = acc + grav_vector
        z_nrm = np.linalg.norm(z, axis=1, keepdims=True)
        z /= z_nrm

        dz = -np.cross(z, np.cross(z, jer, axis=1), axis=1) / z_nrm
        inputs[:, 0] = np.sum(z * (vehicle_mass * (acc + grav_vector)), axis=1)

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
            (
                dz[:, 0] * s_psi
                - dz[:, 1] * c_psi
                - (z[:, 0] * s_psi - z[:, 1] * c_psi) * omg_term
            ),
            (
                dz[:, 0] * c_psi
                + dz[:, 1] * s_psi
                - (z[:, 0] * c_psi + z[:, 1] * s_psi) * omg_term
            ),
            (z[:, 1] * dz[:, 0] - z[:, 0] * dz[:, 1]) / omg_den + yaw_rate,
        ]
    )

    return attitude, inputs


def compute_quadrotor_trajectory(
    polys, t_sample, vehicle_mass, yaw=None, yaw_rate=None, drag_params=None
):

    trajectory_derivatives = compute_trajectory_derivatives(polys, t_sample, 4)
    len_traj = len(t_sample)
    if yaw is not None:
        if yaw == "velocity":
            yaw = np.arctan2(
                trajectory_derivatives[1, :, 1], trajectory_derivatives[1, :, 0]
            )
        yaw = np.broadcast_to(yaw, [len_traj])
    else:
        yaw = np.zeros(len_traj)

    if yaw_rate is not None:
        yaw_rate = np.broadcast_to(yaw_rate, [len_traj])
    else:
        yaw_rate = np.zeros(len_traj)

    attitudes, inputs = flat_output_to_quadrotor_trajectory(
        trajectory_derivatives, vehicle_mass, yaw, yaw_rate, drag_params
    )
    positions = trajectory_derivatives[0, :, :]
    velocities = trajectory_derivatives[1, :, :]

    return QuadrotorTrajectory(np.hstack([positions, attitudes, velocities]), inputs)


def generate_trajectories(
    references,
    degree,
    derivative_weights,
    *,
    continuity_order=3,
    algorithm="closed-form",
    optimize_options=None,
):

    if (derivative_weights < 0.0).any():
        raise ValueError(
            "Weights on derivatives must be a 1D list of nonnegative numbers"
        )
    n_derivs = len(derivative_weights)
    if n_derivs != degree:
        raise ValueError(
            f"Number of derivative masks {n_derivs} != order of the polynomial {degree}"
        )

    t_ref, refs = _parse_references(references)
    durations = np.diff(t_ref)

    poly_dim = PolynomialSize(
        n_poly=refs.shape[0] - 1, n_cfs=degree + 1, dim=refs.shape[2]
    )

    if algorithm == "constrained":
        solver = _solve_constrained
    elif algorithm == "closed-form":
        solver = _solve_closed_form
    else:
        raise ValueError("Unrecognized algorithm")
    polys = solver(
        refs,
        durations,
        poly_dim,
        derivative_weights,
        continuity_order,
        optimize_options,
    )
    return PiecewisePolynomialTrajectory(t_ref, durations, polys)


def _nd_polyvals(coeffs, time, r):
    n_cfs = coeffs.shape[0]  # Coefficients per-piece
    if r == 0:
        return (time ** np.arange(0, n_cfs)) @ coeffs
    n_seq = np.arange(r, n_cfs, dtype=np.int64)
    r_seq = np.arange(0, r, dtype=np.int64)
    return time ** (n_seq - r) @ (
        np.prod(n_seq[None, :] - r_seq[:, None], axis=0)[..., None] * coeffs[n_seq, :]
    )


def _parse_references(references):
    len_traj = len(references)
    t_ref = []

    refs = []
    for idx, it in enumerate(references):
        t_ref.append(it.time)
        pos = it.position
        dim = len(pos)
        placeholder = (
            np.zeros(dim) if idx in (0, len_traj - 1) else np.full([dim], np.nan)
        )

        vel = it.velocity if it.velocity is not None else placeholder
        acc = it.acceleration if it.acceleration is not None else placeholder

        refs.append(np.array([pos, vel, acc]))

    refs = np.asarray(refs)

    return np.asarray(t_ref), refs


def _solve_closed_form(
    refs,
    durations,
    poly_dim,
    derivative_weights,
    r_cts,
    optimize_options,
):
    if r_cts > 3:
        warnings.warn(
            "Constraining continuity above the 3rd order (acceleration) "
            "When using the closed-form algorithm is not well tested"
        )

    if optimize_options is not None:
        warnings.warn(
            "Solving the trajectory generation problem in closed form."
            "Optimizer options will be ignored"
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
            A[r_cts * 2 * i + r, s] = (
                _compute_tvec(poly_dim.n_cfs, r, 0) / durations[i] ** r
            )
            A[r_cts * (2 * i + 1) + r, s] = (
                _compute_tvec(poly_dim.n_cfs, r, 1) / durations[i] ** r
            )

    M = np.zeros((poly_dim.n_poly * 2 * r_cts, r_cts * (poly_dim.n_poly + 1)))
    for i in range(poly_dim.n_poly):
        s1 = np.s_[2 * r_cts * i : 2 * r_cts * (i + 1)]
        s2 = np.s_[r_cts * i : r_cts * (i + 2)]
        M[s1, s2] = np.eye(2 * r_cts)

    num_d = r_cts * (poly_dim.n_poly + 1)

    # compute C
    C = np.eye(num_d)
    # fix all pos(poly_dim.n_poly+1) + start va(2) +  va(2)
    fix_idx = np.concatenate([np.arange(0, num_d, r_cts), [1, 2, num_d - 2, num_d - 1]])
    free_idx = np.setdiff1d(np.arange(num_d), fix_idx)
    C = np.hstack([C[:, fix_idx], C[:, free_idx]])

    AiMC = la.lstsq(A, M @ C)[0]
    R = AiMC.T @ Q @ AiMC

    n_fix = fix_idx.size
    Rpp = R[n_fix:, n_fix:]
    Rfp = R[:n_fix, n_fix:]

    for d in range(poly_dim.dim):

        ref = refs[..., d]
        df = np.concatenate([ref[:, 0], ref[0, 1:3], ref[-1, 1:3]])
        dp = -la.solve(Rpp, Rfp.T @ df)

        coeffs = np.reshape(AiMC @ np.concatenate([df, dp]), poly_dim[0:2])
        poly_coeffs[:, :, d] = (1.0 / durations[..., None]) ** np.arange(
            0, poly_dim.n_cfs
        ) * coeffs

    return poly_coeffs


def _solve_constrained(
    refs,
    durations,
    poly_dim,
    derivative_weights,
    r_cts,
    optimize_options,
):

    opts = {"method": "SLSQP", "tol": 1e-10}
    if optimize_options is not None:
        opts.update(optimize_options)
    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    Q = np.zeros((n_vars, n_vars))
    for r, c_r in enumerate(derivative_weights):
        Q += c_r * la.block_diag(*_compute_Q(poly_dim.n_cfs, r, durations))

    poly_coeffs = np.zeros(poly_dim)
    Aeq_1, beq_1 = _compute_continuity_constraints(poly_dim, durations, r_cts)
    for d in range(poly_dim.dim):
        Aeq_0, beq_0 = _compute_dynamical_constraints(
            poly_dim, refs[:, :, d], durations
        )

        Aeq = np.vstack([Aeq_0, Aeq_1])
        beq = np.concatenate([beq_0, beq_1])

        constr = optimize.LinearConstraint(Aeq, beq, beq)  # type: ignore
        soln = optimize.minimize(
            lambda x: (x @ Q @ x) / 2,
            np.zeros(n_vars),
            constraints=constr,
            jac=lambda x: Q @ x,
            **opts,
        )
        coeffs = soln.x.reshape(poly_dim[0:2])
        poly_coeffs[:, :, d] = (1.0 / durations[..., None]) ** np.arange(
            0, poly_dim.n_cfs
        ) * coeffs
    return poly_coeffs


def _compute_continuity_constraints(poly_dim, durations, r_cts):

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


def _compute_dynamical_constraints(poly_dim, refs, durations):

    n_vars = poly_dim.n_poly * poly_dim.n_cfs
    n_constrain_orders = np.count_nonzero(~np.isnan(refs), axis=1)
    Aeq = np.zeros((n_constrain_orders.sum(), n_vars))
    beq = np.zeros(n_constrain_orders.sum())

    row_its = np.concatenate([[0], n_constrain_orders.cumsum()])
    for i in range(poly_dim.n_poly + 1):
        idx, tau = (i - 1, 1.0) if i == poly_dim.n_poly else (i, 0.0)
        s = np.s_[poly_dim.n_cfs * idx : poly_dim.n_cfs * (1 + idx)]
        for r in range(n_constrain_orders[i]):
            Aeq[row_its[i] + r, s] = (
                _compute_tvec(poly_dim.n_cfs, r, tau) / durations[idx] ** r
            )
            beq[row_its[i] + r] = refs[i, r]
    return Aeq, beq


def _compute_Q(n_cfs, r, tau):
    Q = np.zeros((len(tau), n_cfs, n_cfs))

    i, l = np.meshgrid(*[np.arange(r, n_cfs)] * 2, sparse=True)
    m_seq = np.arange(0, r)[:, None, None]
    k = -2 * r + 1
    Q[:, i, l] = (
        np.prod((i - m_seq) * (l - m_seq), axis=0)
        / (k + i + l)
        * tau[:, None, None] ** k
    )
    return Q


def _compute_tvec(n_cfs, r, tau):
    tvec = np.zeros(n_cfs)
    n_seq = np.arange(r, n_cfs)
    r_seq = np.arange(0, r)[:, None]
    tvec[n_seq] = np.prod(n_seq - r_seq, axis=0) * tau ** (n_seq - r)
    return tvec
