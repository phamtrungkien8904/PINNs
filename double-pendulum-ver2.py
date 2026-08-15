import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


# ============================================================
# DOUBLE PENDULUM
# Finite Difference Method applied directly to
# the implicit Euler-Lagrange equations.
#
# No explicit formulas for theta1_ddot or theta2_ddot.
# No odeint / solve_ivp.
# ============================================================


# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
m1 = 1.0          # kg
m2 = 1.0          # kg

l1 = 1.0          # m
l2 = 1.0          # m

g = 9.81          # m/s^2


# ------------------------------------------------------------
# Time discretization
# ------------------------------------------------------------
t0 = 0.0
t_end = 10.0

dt = 0.001

t = np.arange(t0, t_end + dt, dt)
N = len(t)


# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------
theta1_0 = np.deg2rad(-9.0)
theta2_0 = np.deg2rad(9.0)

omega1_0 = 0.0
omega2_0 = 0.0


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------
theta1 = np.zeros(N)
theta2 = np.zeros(N)

theta1[0] = theta1_0
theta2[0] = theta2_0


# ============================================================
# Euler-Lagrange residual
# ============================================================
def lagrange_residual(
        theta_next,
        theta_now,
        theta_prev,
        dt):

    """
    Residual of the two implicit Euler-Lagrange equations.

    Unknown:
        theta_next = [theta1_(n+1), theta2_(n+1)]

    Known:
        theta_now  = [theta1_n, theta2_n]
        theta_prev = [theta1_(n-1), theta2_(n-1)]
    """

    th1_next, th2_next = theta_next
    th1, th2 = theta_now
    th1_prev, th2_prev = theta_prev

    # --------------------------------------------------------
    # Finite-difference velocity at time n
    #
    #           theta(n+1) - theta(n-1)
    # theta' = --------------------------
    #                    2 dt
    # --------------------------------------------------------
    w1 = (th1_next - th1_prev) / (2.0 * dt)
    w2 = (th2_next - th2_prev) / (2.0 * dt)

    # --------------------------------------------------------
    # Finite-difference acceleration at time n
    #
    #           theta(n+1) - 2 theta(n) + theta(n-1)
    # theta'' = --------------------------------------------
    #                            dt^2
    # --------------------------------------------------------
    a1 = (
        th1_next
        - 2.0 * th1
        + th1_prev
    ) / dt**2

    a2 = (
        th2_next
        - 2.0 * th2
        + th2_prev
    ) / dt**2

    delta = th1 - th2

    # ========================================================
    # Euler-Lagrange equation 1
    # ========================================================
    F1 = (
        (m1 + m2) * l1**2 * a1

        + m2 * l1 * l2
        * a2
        * np.cos(delta)

        + m2 * l1 * l2
        * w2**2
        * np.sin(delta)

        + (m1 + m2)
        * g * l1
        * np.sin(th1)
    )

    # ========================================================
    # Euler-Lagrange equation 2
    # ========================================================
    F2 = (
        m2 * l2**2 * a2

        + m2 * l1 * l2
        * a1
        * np.cos(delta)

        - m2 * l1 * l2
        * w1**2
        * np.sin(delta)

        + m2
        * g * l2
        * np.sin(th2)
    )

    return np.array([F1, F2])


# ============================================================
# FIRST TIME STEP
# ============================================================
#
# Central difference normally needs:
#
#       theta[-1], theta[0], theta[1]
#
# but theta[-1] does not exist.
#
# Use the initial velocity:
#
# theta_dot(0) =
#       [theta(1) - theta(-1)] / (2 dt)
#
# Therefore
#
# theta(-1) = theta(1) - 2 dt omega(0)
#
# We substitute this into the Euler-Lagrange equations
# and solve directly for theta(1).
# ============================================================

theta_initial = np.array([
    theta1_0,
    theta2_0
])

omega_initial = np.array([
    omega1_0,
    omega2_0
])


def first_step_residual(theta_next):

    # Ghost point from prescribed initial velocity
    theta_prev = (
        theta_next
        - 2.0 * dt * omega_initial
    )

    return lagrange_residual(
        theta_next,
        theta_initial,
        theta_prev,
        dt
    )


# Simple initial guess
guess = theta_initial + dt * omega_initial

solution = root(
    first_step_residual,
    guess,
    method="hybr"
)

if not solution.success:
    raise RuntimeError(
        "First timestep did not converge: "
        + solution.message
    )

theta1[1] = solution.x[0]
theta2[1] = solution.x[1]


# ============================================================
# MAIN FDM TIME LOOP
# ============================================================

for n in range(1, N - 1):

    theta_now = np.array([
        theta1[n],
        theta2[n]
    ])

    theta_prev = np.array([
        theta1[n - 1],
        theta2[n - 1]
    ])

    # --------------------------------------------------------
    # Predictor for theta_(n+1)
    #
    # Linear extrapolation:
    #
    # theta_(n+1) ≈ 2 theta_n - theta_(n-1)
    #
    # This is ONLY the initial guess for Newton/root.
    # It is NOT the final solution.
    # --------------------------------------------------------
    guess = (
        2.0 * theta_now
        - theta_prev
    )

    # --------------------------------------------------------
    # Solve:
    #
    # F1(theta1_(n+1), theta2_(n+1)) = 0
    # F2(theta1_(n+1), theta2_(n+1)) = 0
    #
    # directly.
    # --------------------------------------------------------
    solution = root(
        lagrange_residual,
        guess,
        args=(
            theta_now,
            theta_prev,
            dt
        ),
        method="hybr",
        tol=1e-10
    )

    if not solution.success:
        raise RuntimeError(
            f"Solver failed at step {n}, "
            f"t = {t[n]:.6f} s\n"
            f"{solution.message}"
        )

    theta1[n + 1] = solution.x[0]
    theta2[n + 1] = solution.x[1]


# ============================================================
# Calculate velocities afterward
# only for plotting / diagnostics
# ============================================================

omega1 = np.gradient(
    theta1,
    dt,
    edge_order=2
)

omega2 = np.gradient(
    theta2,
    dt,
    edge_order=2
)


# ============================================================
# Cartesian coordinates
# ============================================================

x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)

x2 = (
    l1 * np.sin(theta1)
    + l2 * np.sin(theta2)
)

y2 = (
    -l1 * np.cos(theta1)
    - l2 * np.cos(theta2)
)


# ============================================================
# Energy
# ============================================================

delta = theta1 - theta2

T = (
    0.5 * (m1 + m2) * l1**2 * omega1**2
    + 0.5 * m2 * l2**2 * omega2**2
    + m2 * l1 * l2
    * omega1 * omega2
    * np.cos(delta)
)

V = (
    -(m1 + m2)
    * g * l1
    * np.cos(theta1)

    - m2
    * g * l2
    * np.cos(theta2)
)

E = T + V


# ============================================================
# Plot theta1 and theta2
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(
    t,
    np.rad2deg(theta1),
    label=r"$\theta_1$"
)

plt.plot(
    t,
    np.rad2deg(theta2),
    label=r"$\theta_2$"
)

plt.xlabel("Time [s]")
plt.ylabel("Angle [degree]")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# Phase plots
# ============================================================

plt.figure(figsize=(6, 5))

plt.plot(
    theta1,
    omega1,
    label=r"$\theta_1$"
)

plt.plot(
    theta2,
    omega2,
    label=r"$\theta_2$"
)

plt.xlabel(r"$\theta$ [rad]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# Bob trajectories
# ============================================================

plt.figure(figsize=(6, 6))

plt.plot(
    x1,
    y1,
    label="Mass 1"
)

plt.plot(
    x2,
    y2,
    label="Mass 2"
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.axis("equal")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# Energy conservation
# ============================================================

plt.figure(figsize=(8, 4))

plt.plot(
    t,
    E - E[0]
)

plt.xlabel("Time [s]")
plt.ylabel(r"$E(t)-E(0)$ [J]")

plt.grid()
plt.tight_layout()
plt.show()


# ============================================================
# Print results
# ============================================================

print("FDM simulation finished.")
print(f"Number of time points : {N}")
print(f"dt                    : {dt}")
print()
print("Final values:")
print(f"theta1 = {theta1[-1]:.8f} rad")
print(f"theta2 = {theta2[-1]:.8f} rad")
print(f"omega1 = {omega1[-1]:.8f} rad/s")
print(f"omega2 = {omega2[-1]:.8f} rad/s")

relative_energy_error = (
    np.max(np.abs(E - E[0]))
    / max(abs(E[0]), 1e-15)
)

print()
print(
    "Maximum relative energy error = "
    f"{relative_energy_error:.3e}"
)