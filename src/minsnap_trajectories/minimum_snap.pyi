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

from typing import Any, Dict, Literal, NamedTuple, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

class Waypoint:
    """This class holds a waypoint consisting of position and optionally velocity,
    acceleration, etc. values that defines a piecewise polynomial trajectory.
    """

    def __init__(
        self,
        time: float,
        position: ArrayLike,
        velocity: Optional[ArrayLike] = None,
        acceleration: Optional[ArrayLike] = None,
        jerk: Optional[ArrayLike] = None,
        snap: Optional[ArrayLike] = None,
    ):
        """Creates a waypoint

        Parameters
        ----------
        time : float
            The timestamp of the waypoint
        position : ArrayLike
            The position of the waypoint
        velocity : Optional[ArrayLike], optional
            The velocity of the waypoint, by default None
        acceleration : Optional[ArrayLike], optional
            The acceleration at this waypoint, by default None
        jerk : Optional[ArrayLike], optional
            The jerk (3rd-derivative of position) at the waypoint; Not commonly
            specified, by default None
        snap : Optional[ArrayLike], optional
            The snap (4th-derivative of position) at the waypoint; Not commonly
            specified, by default None
        """

    @property
    def position(self) -> NDArray:
        """Accesses the waypoint position"""

    @property
    def velocity(self) -> NDArray:
        """Accesses the velocity at this waypoint"""

    @property
    def acceleration(self) -> NDArray:
        """Accesses the acceleration at this waypoint"""

class PiecewisePolynomialTrajectory(NamedTuple):
    """A piecewise polynomial trajectory, defined by an array of timestamps
    (alternatively an array of durations between each pair of timestamps) and polynomial
    coefficients
    """

    time_reference: NDArray[np.float64]
    durations: NDArray[np.float64]
    coefficients: NDArray[np.float64]

class RotorDragParameters(NamedTuple):
    """Parameters for the RDR rotor drag model.
    cp is a user defined time-constant for filtering the rotor drag as a function of
    quadrotor velocity
    dh, dv are diagonal terms of the D rotor drag matrix, D = diag([dh, dv, dh])
    """

    cp: float
    dh: float
    dv: float

class QuadrotorTrajectory(NamedTuple):
    """State/input trajectory of a quadrotor

    state is a horizontal stack [position, attitude, velocity], where attitude are
    unit-quaternions

    input is a horizontal stack [thrust, body_rates]
    """

    state: NDArray[np.float64]
    input: NDArray[np.float64]

    position: NDArray[np.float64]
    attitude: NDArray[np.float64]
    velocity: NDArray[np.float64]

    thrust: NDArray[np.float64]
    body_rates: NDArray[np.float64]

def generate_trajectory(
    references: Sequence[Waypoint],
    degree: int,
    *,
    minimized_orders: Union[int, Sequence[int]] = 4,
    continuous_orders: int = 3,
    algorithm: Literal["closed-form", "constrained"] = "closed-form",
    optimize_options: Optional[Dict[str, Any]] = None,
) -> PiecewisePolynomialTrajectory:
    """Plans a piecewise-polynomial trajectory

    Parameters
    ----------
    references : Sequence[Waypoint]
        A sequence of waypoints defining the trajectory
    degree : int
        The degree of the piecewise polynomial
    minimized_orders : Union[int, Sequence[int]], optional
        The order of derivatives of position to be minimized, by default 4
    continuous_orders : int, optional
        The number of orders of derivatives constrained to be continuous, by default 3
    algorithm : Literal["closed-form", "constrained"], optional
        The algorithm to use, either:

            closed-form: Optimize end-derivatives per Bry and Roy, making for an
            unconstrained optimization problem that can be solved in closed form

            constrained: Directly optimize polynomial coefficients per Mellinger and
            Kumar, requiring a numerical solver

        default "closed-form"
    optimize_options : Optional[Dict[str, Any]], optional
        Options to pass to scipy.optimize.minimize; if the closed-form algorithm is
        used, this argument is ignored, by default None

    Returns
    -------
    PiecewisePolynomialTrajectory
        A namedtuple holding the waypoint timestamps, the duration for transit between
        waypoints, and the piecewise polynomial coefficients
    """

def compute_trajectory_derivatives(
    polys: PiecewisePolynomialTrajectory,
    t_sample: ArrayLike,
    order: int,
) -> NDArray[np.float64]:
    """Samples a piecewise polynomial trajectory at given times to give concrete
    trajectory derivatives, i.e. position/velocity/acceleration/etc...

    Parameters
    ----------
    polys : PiecewisePolynomialTrajectory
        A piecewise polynomial created by `generate_trajectory`
    t_sample : ArrayLike
        An array of sample times
    order : int
        The order of trajectory derivatives to complete

    Returns
    -------
    NDArray
        Trajectory derivatives stacked for each requested order along the first axis,
        making for an array with size order x len(time_sample) x dimension of polynomial
    """

def compute_quadrotor_trajectory(
    polys: PiecewisePolynomialTrajectory,
    t_sample: ArrayLike,
    vehicle_mass: float,
    yaw: Optional[Union[ArrayLike, Literal["velocity"]]] = None,
    yaw_rate: Optional[ArrayLike] = None,
    drag_params: Optional[RotorDragParameters] = None,
) -> QuadrotorTrajectory:
    """Computes a trajectory of quadrotor states and inputs at given times

    Parameters
    ----------
    polys : PiecewisePolynomialTrajectory
        A piecewise polynomial trajectory (must be 3D) created by `generate_trajectory`
    t_sample : ArrayLike
        An array of sample times
    vehicle_mass : float
        Mass of the quadrotor to generate trajectories for
    yaw : Optional[Union[ArrayLike, Literal[&quot;velocity&quot;]]], optional
        Yaw angle references (not planned by the minimum-snap algorithm).

        If left unspecified, the yaw angle along the trajectory will be set to 0.

        If specified "velocity", the yaw angle will align the quadrotor to the velocity
        at each point along the trajectory, i.e. yaw = atan2(velocity_y, velocity.x).

        By default None
    yaw_rate : Optional[ArrayLike], optional
        Yaw rate references (not planned by the minimum-snap algorithm). If left unspecified, the yaw rate along the trajectory will be set to 0, by default None
    drag_params : Optional[RotorDragParameters], optional
        Parameters of the RDR rotor drag model, by default None

    Returns
    -------
    QuadrotorTrajectory
        A namedtuple holding the quadrotor states and inputs
    """
