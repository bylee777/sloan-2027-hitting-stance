"""Constant-acceleration (Statcast 9P) trajectory utilities.

Statcast coordinates (ft): x catcher's-right (+ = 1B side), y from the back tip of home plate
toward the pitcher, z up. vx0..vz0 / ax..az are the fit's velocity and (constant)
acceleration at y = 50 ft. plate_x / plate_z are the crossing at the FRONT of the plate
(y = 17/12 ft) — validated below by reproducing pfx_x / pfx_z.
"""

import numpy as np
import pandas as pd

G = 32.174  # ft/s^2
Y_FRONT = 17 / 12  # front edge of home plate, ft


def t_at_y(df, y):
    """Time (s) after the y=50 reference at which the ball reaches depth y (ft). Negative if y > 50."""
    vy0, ay = df["vy0"].to_numpy(), df["ay"].to_numpy()
    y = np.asarray(y, dtype=float)
    disc = vy0**2 - 2 * ay * (50.0 - y)
    return (-vy0 - np.sqrt(disc)) / ay


def state_at_y(df, y):
    """Position/velocity at depth y, anchored on the measured plate crossing (not on x0/z0)."""
    tp = t_at_y(df, Y_FRONT)
    t = t_at_y(df, y)
    vx0, vy0, vz0 = (df[c].to_numpy() for c in ("vx0", "vy0", "vz0"))
    ax, ay, az = (df[c].to_numpy() for c in ("ax", "ay", "az"))
    x = df["plate_x"].to_numpy() + vx0 * (t - tp) + 0.5 * ax * (t**2 - tp**2)
    z = df["plate_z"].to_numpy() + vz0 * (t - tp) + 0.5 * az * (t**2 - tp**2)
    return dict(t=t, x=x, z=z, vx=vx0 + ax * t, vy=vy0 + ay * t, vz=vz0 + az * t)


def spin_accel(df):
    """Non-gravity, non-drag ('induced') acceleration components, ft/s^2."""
    return df["ax"].to_numpy(), df["az"].to_numpy() + G
