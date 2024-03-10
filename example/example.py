"""
Copyright © 2024 Hs293Go

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

import matplotlib.pyplot as plt
import numpy as np

import minsnap_trajectories as ms


def main():
    refs = [
        ms.Waypoint(
            time=0.0,
            position=np.array([0.0, 0.0, 10.0]),
        ),
        ms.Waypoint(  # Any higher-order derivatives
            time=8.0,
            position=np.array([10.0, 0.0, 10.0]),
            velocity=np.array([0.0, 5.0, 0.0]),
            acceleration=np.array([0.1, 0.0, 0.0]),
        ),
        ms.Waypoint(  # Potentially leave intermediate-order derivatives unspecified
            time=16.0,
            position=np.array([20.0, 0.0, 10.0]),
            jerk=np.array([0.1, 0.0, 0.2]),
        ),
    ]

    polys = ms.generate_trajectory(
        refs,
        degree=8,  # Polynomial degree
        idx_minimized_orders=(3, 4),  # Minimize derivatives in these orders (>= 2)
        num_continuous_orders=3,  # Constrain continuity of derivatives up to order (>= 3)
        algorithm="closed-form",  # Or "constrained"
    )

    t = np.linspace(0, 16, 100)
    #  Sample up to the 3rd order (acceleration) -----v
    pva = ms.compute_trajectory_derivatives(polys, t, 3)
    position, *_ = pva

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4))
    ax.plot(position[:, 0], position[:, 1], position[:, 2], label="Position Trajectory")

    position_waypoints = np.array([it.position for it in refs])
    ax.plot(
        position_waypoints[:, 0],
        position_waypoints[:, 1],
        position_waypoints[:, 2],
        "ro",
        label="Position Waypoints",
    )
    ax.quiver(
        *refs[1].position,
        *refs[1].velocity,
        color="g",
        label="Velocity specified at waypoint 1",
    )
    ax.set_zlim(8, 12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    try:
        fig.savefig("example/minsnap_trajectories_example.png")
    except FileNotFoundError:
        plt.show()


if __name__ == "__main__":
    main()
