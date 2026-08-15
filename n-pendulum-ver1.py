import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root


# ============================================================
# N-PENDULUM
# Finite Difference Method directly on Euler-Lagrange equations
#
# No explicit theta_ddot = f(theta, omega)
# No solve_ivp
# No odeint
# ============================================================


# ============================================================
# USER PARAMETERS
# ============================================================

# Number of pendulums
n_pend = 5

# Mass of each bob
m = np.ones(n_pend)

# Length of each rod
l = np.ones(n_pend)

g = 9.81


# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

theta0 = np.deg2rad([
    120.0,
    -10.0,
    20.0,
    30.0,
    -20.0
])

omega0 = np.zeros(n_pend)


# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

t0 = 0.0
t_end = 10.0

dt = 0.001

t = np.arange(t0, t_end + dt, dt)

Nt = len(t)


# ============================================================
# Sanity checks
# ============================================================

assert len(m) == n_pend
assert len(l) == n_pend
assert len(theta0) == n_pend
assert len(omega0) == n_pend


# ============================================================
# PRECOMPUTE MASS COUPLING MATRIX A
#
# A_ij =
#
#       l_i l_j sum_{k=max(i,j)}^N m_k
#
# ============================================================

# tail_mass[i] =
#
# m_i + m_(i+1) + ... + m_N

tail_mass = np.cumsum(m[::-1])[::-1]


A = np.zeros((n_pend, n_pend))

for i in range(n_pend):

    for j in range(n_pend):

        k = max(i, j)

        A[i, j] = (
            l[i]
            * l[j]
            * tail_mass[k]
        )


print("Tail masses:")
print(tail_mass)

print("\nA matrix:")
print(A)


# ============================================================
# LAGRANGIAN / EULER-LAGRANGE RESIDUAL
# ============================================================

def lagrange_residual(
        theta_next,
        theta_now,
        theta_prev,
        dt
    ):

    """
    Direct finite-difference Euler-Lagrange residual.

    Parameters
    ----------
    theta_next:
        theta at t_(n+1), UNKNOWN

    theta_now:
        theta at t_n

    theta_prev:
        theta at t_(n-1)

    Returns
    -------
    F:
        N Euler-Lagrange residuals

        F_i = 0
    """

    # ========================================================
    # Central finite-difference velocity
    #
    #              theta_(n+1) - theta_(n-1)
    # theta_dot = -----------------------------
    #                         2 dt
    # ========================================================

    omega = (
        theta_next
        - theta_prev
    ) / (2.0 * dt)


    # ========================================================
    # Central finite-difference acceleration
    #
    #               theta_(n+1)
    #             - 2 theta_n
    #             + theta_(n-1)
    #
    # theta_ddot = -------------------
    #                    dt^2
    # ========================================================

    alpha = (
        theta_next
        - 2.0 * theta_now
        + theta_prev
    ) / dt**2


    # ========================================================
    # theta_i - theta_j matrix
    #
    # delta[i,j] = theta_i - theta_j
    # ========================================================

    delta = (
        theta_now[:, None]
        - theta_now[None, :]
    )


    # ========================================================
    # Mass matrix
    #
    # M_ij(theta) =
    #
    # A_ij cos(theta_i - theta_j)
    # ========================================================

    M = A * np.cos(delta)


    # ========================================================
    # Velocity nonlinear term
    #
    # C_i =
    #
    # sum_j
    #
    # A_ij sin(theta_i - theta_j)
    #       theta_dot_j^2
    # ========================================================

    C = (
        A * np.sin(delta)
    ) @ (omega**2)


    # ========================================================
    # Gravity
    #
    # G_i =
    #
    # (sum_{k=i}^N m_k)
    #       g l_i sin(theta_i)
    # ========================================================

    G = (
        tail_mass
        * g
        * l
        * np.sin(theta_now)
    )


    # ========================================================
    # Euler-Lagrange equations
    #
    # M(theta) theta_ddot
    # +
    # C(theta, theta_dot)
    # +
    # G(theta)
    # =
    # 0
    # ========================================================

    F = (
        M @ alpha
        + C
        + G
    )

    return F


# ============================================================
# STORAGE
# ============================================================

# Shape:
#
# time x pendulum
#
# theta[n, i]
#
# = angle of pendulum i at timestep n

theta = np.zeros(
    (Nt, n_pend)
)

theta[0, :] = theta0


# ============================================================
# FIRST TIME STEP
# ============================================================
#
# Central differences require:
#
# theta[-1], theta[0], theta[1]
#
# But theta[-1] does not physically exist.
#
# Use
#
# theta_dot(0) =
#
# theta[1] - theta[-1]
# -----------------------
#        2 dt
#
# therefore
#
# theta[-1] =
#
# theta[1] - 2 dt omega0
#
# ============================================================

def first_step_residual(theta_next):

    theta_prev = (
        theta_next
        - 2.0 * dt * omega0
    )

    return lagrange_residual(
        theta_next,
        theta0,
        theta_prev,
        dt
    )


# Initial predictor
guess = (
    theta0
    + dt * omega0
)


solution = root(
    first_step_residual,
    guess,
    method="hybr",
    tol=1e-10
)


if not solution.success:

    raise RuntimeError(
        "First step failed:\n"
        + solution.message
    )


theta[1, :] = solution.x


# ============================================================
# MAIN FDM LOOP
# ============================================================

for n in range(1, Nt - 1):

    theta_prev = theta[n - 1, :]
    theta_now = theta[n, :]


    # --------------------------------------------------------
    # Predictor
    #
    # Constant velocity extrapolation
    #
    # theta_(n+1) ~
    #
    # 2 theta_n - theta_(n-1)
    # --------------------------------------------------------

    guess = (
        2.0 * theta_now
        - theta_prev
    )


    # --------------------------------------------------------
    # Solve N nonlinear equations
    #
    # F_1 = 0
    # F_2 = 0
    # ...
    # F_N = 0
    #
    # Unknown:
    #
    # theta_1^(n+1)
    # theta_2^(n+1)
    # ...
    # theta_N^(n+1)
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
            f"Solver failed at timestep {n}\n"
            f"t = {t[n]:.6f} s\n"
            + solution.message
        )


    theta[n + 1, :] = solution.x


# ============================================================
# VELOCITY
#
# Only calculated afterward for diagnostics/plotting.
# It is NOT used as an independent ODE state.
# ============================================================

omega = np.gradient(
    theta,
    dt,
    axis=0,
    edge_order=2
)


# ============================================================
# CALCULATE x,y POSITIONS
# ============================================================
#
# x_k =
#
# sum_{j=1}^k l_j sin(theta_j)
#
#
# y_k =
#
# -sum_{j=1}^k l_j cos(theta_j)
#
# ============================================================

x = np.cumsum(
    l[None, :]
    * np.sin(theta),
    axis=1
)

y = np.cumsum(
    -l[None, :]
    * np.cos(theta),
    axis=1
)


# ============================================================
# ENERGY
# ============================================================

energy = np.zeros(Nt)


for n in range(Nt):

    delta = (
        theta[n, :, None]
        - theta[n, None, :]
    )

    M = (
        A * np.cos(delta)
    )


    # --------------------------------------------------------
    # Kinetic energy
    #
    # T = 1/2 omega^T M omega
    # --------------------------------------------------------

    T = (
        0.5
        * omega[n]
        @ M
        @ omega[n]
    )


    # --------------------------------------------------------
    # Potential energy
    #
    # V =
    #
    # - sum_i
    #
    # mu_i g l_i cos(theta_i)
    # --------------------------------------------------------

    V = -np.sum(
        tail_mass
        * g
        * l
        * np.cos(theta[n])
    )


    energy[n] = T + V


# ============================================================
# PLOT ALL ANGLES
# ============================================================

plt.figure(figsize=(10, 5))

for i in range(n_pend):

    plt.plot(
        t,
        np.rad2deg(theta[:, i]),
        label=fr"$\theta_{i+1}$"
    )


plt.xlabel("Time [s]")
plt.ylabel("Angle [degree]")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# PLOT TRAJECTORY OF EVERY BOB
# ============================================================

plt.figure(figsize=(7, 7))

for i in range(n_pend):

    plt.plot(
        x[:, i],
        y[:, i],
        label=f"Mass {i+1}"
    )


plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.axis("equal")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# ENERGY CONSERVATION
# ============================================================

plt.figure(figsize=(10, 4))

plt.plot(
    t,
    energy - energy[0]
)

plt.xlabel("Time [s]")

plt.ylabel(
    r"$E(t)-E(0)$ [J]"
)

plt.grid()

plt.tight_layout()
plt.show()


# ============================================================
# PRINT
# ============================================================

print("\nSimulation completed.")

print(
    f"N pendulums      = {n_pend}"
)

print(
    f"Timesteps        = {Nt}"
)

print(
    f"dt               = {dt}"
)

print("\nFinal angles:")

for i in range(n_pend):

    print(
        f"theta_{i+1} = "
        f"{theta[-1, i]: .8f} rad"
    )