import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Custom settings
plt.style.use('classic')

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'''
    \usepackage[T1]{fontenc}
    \usepackage{lmodern}
    \usepackage[utf8]{inputenc}
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{siunitx}
    \usepackage{sfmath}
    '''
})
plt.rcParams.update({
    # Figure settings
    'figure.dpi': 300,
    # 'figure.figsize': (10/2.54, 6/2.54),  # 10x6 cm in inches (1 figure per line)
    # 'figure.figsize': (8/2.54, 6/2.54),  # 10x6 cm in inches (2 figures per line)
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'axes.labelcolor': 'black',
    'savefig.facecolor': 'white',
    'font.family': 'sans-serif',
    'font.sans-serif': 'Arial',
    # 'mathtext.fontset': 'cm',
    'figure.constrained_layout.use': True,

    # Ticks
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.minor.width":0,
    "ytick.minor.width": 0,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,  

    # Legend
    'legend.frameon': False,
    'legend.title_fontsize': 8,
    'legend.fontsize': 8,
    'legend.handlelength': 2,
    'legend.loc': 'best',
    'legend.numpoints': 1,

    # Line style
    'lines.linestyle': '-',
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'lines.markeredgecolor': 'white',
    'lines.markeredgewidth': 0.5,
})

# Pendulum parameters
L1, L2 = 1.0, 1.0
m1, m2 = 1.0, 1.0
g = 10

def deriv(y, t, L1, L2, m1, m2):
    theta1, theta1dot, theta2, theta2dot = y
    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)
    denom = m1 + m2 * s**2

    theta1dotdot = (
        m2 * g * np.sin(theta2) * c
        - m2 * s * (L1 * theta1dot**2 * c + L2 * theta2dot**2)
        - (m1 + m2) * g * np.sin(theta1)
    ) / (L1 * denom)

    theta2dotdot = (
        (m1 + m2) * (L1 * theta1dot**2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c)
        + m2 * L2 * theta2dot**2 * s * c
    ) / (L2 * denom)

    return theta1dot, theta1dotdot, theta2dot, theta2dotdot

tmax, dt = 10, 0.01
t = np.arange(0, tmax + dt, dt)
y0 = np.array([-np.pi/8, 0, -np.pi/10, 0])
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

theta1, theta2 = y[:, 0], y[:, 2]
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)


# ############### ANIMATION SECTION ##############
# ################################################

# # Animate and export at a true 30 fps so playback duration matches simulation time.
# animation_fps = 30
# animation_t = np.arange(0, tmax + 1 / animation_fps, 1 / animation_fps)
# x1_anim = np.interp(animation_t, t, x1)
# y1_anim = np.interp(animation_t, t, y1)
# x2_anim = np.interp(animation_t, t, x2)
# y2_anim = np.interp(animation_t, t, y2)

# fig, ax = plt.subplots(figsize=(4/2.54, 4/2.54))
# ax.set_xlim(-(L1 + L2 + 0.3), L1 + L2 + 0.3)
# ax.set_ylim(-(L1 + L2 + 0.3), L1 + L2 + 0.3)
# ax.set_aspect("equal")
# ax.axis("off")

# rod, = ax.plot([], [], color="black", lw=1)
# bob1, = ax.plot([], [], "o", color="blue", markersize=5)
# bob2, = ax.plot([], [], "o", color="red", markersize=5)
# trail, = ax.plot([], [], color="red", lw=1.5, alpha=0.5)
# time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=5)

# def init():
#     rod.set_data([], [])
#     bob1.set_data([], [])
#     bob2.set_data([], [])
#     trail.set_data([], [])
#     time_text.set_text("")
#     return rod, bob1, bob2, trail, time_text

# def animate(i):
#     # Fixed data formatting for set_data
#     rod.set_data([0, x1_anim[i], x2_anim[i]], [0, y1_anim[i], y2_anim[i]])
#     bob1.set_data([x1_anim[i]], [y1_anim[i]])
#     bob2.set_data([x2_anim[i]], [y2_anim[i]])

#     start = max(0, i - 80)
#     trail.set_data(x2_anim[start:i + 1], y2_anim[start:i + 1])
#     time_text.set_text(f"t = {animation_t[i]:.2f} s")
#     return rod, bob1, bob2, trail, time_text

# # Match interval to real-time playback (1000 ms / 30 fps ~ 33ms)
# ani = animation.FuncAnimation(
#     fig,
#     animate,
#     frames=len(animation_t),
#     init_func=init,
#     interval=1000 / animation_fps,
#     blit=True,
# )

# # Save animation loop
# ani.save("double_pendulum_animation.gif", writer="pillow", fps=animation_fps)
# print("Saved double_pendulum_animation.gif successfully.")
# plt.show()

############### END ANIMATION SECTION #########################

fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
ax.plot(t, theta1, label="Bob 1", color="blue")
ax.plot(t, theta2, label="Bob 2", color="red")
ax.set_title("Double Pendulum")
ax.set_xlim(0, tmax)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.legend()
plt.savefig("double_pendulum_time.png", dpi=600)
plt.show()


np.savetxt("double_pendulum_data.dat", np.column_stack((t, theta1, theta2)), header="# Time(s) Theta1(rad) Theta2(rad)", comments='')


bins = len(t)  # Number of bins for FFT, power of 2 for efficiency
signal_fft1 = np.fft.fft(theta1, n=bins)
signal_fft2 = np.fft.fft(theta2, n=bins)
freq = np.fft.fftfreq(bins, d=(t[1] - t[0]))
mask = freq > 0
freq = freq[mask]
mag1 = 2.0 / bins * np.abs(signal_fft1[mask])
mag2 = 2.0 / bins * np.abs(signal_fft2[mask])

fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
ax.plot(freq, mag1, label="Bob 1", color="blue", marker="o")
ax.plot(freq, mag2, label="Bob 2", color="red", marker="s")
ax.set_title("Frequency Spectrum")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_xlim(0, 2)  # Limit x-axis to focus on low frequencies
ax.legend()
plt.savefig("double_pendulum_fft.png", dpi=600)
plt.show()


# signal_ifft1 = np.fft.ifft(signal_fft1)
# signal_ifft2 = np.fft.ifft(signal_fft2)

# fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
# ax.plot(t, theta1, label="Original Bob 1", color="blue")
# ax.plot(t, signal_ifft1.real, label="Reconstructed Bob 1", color="cyan", linestyle="--")
# ax.plot(t, theta2, label="Original Bob 2", color="red")
# ax.plot(t, signal_ifft2.real, label="Reconstructed Bob 2", color="orange", linestyle="--")
# ax.set_title("Original and Reconstructed Signals")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Angle (rad)")
# ax.legend()
# plt.savefig("double_pendulum_ifft.png", dpi=600)
# plt.show()