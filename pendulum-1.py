import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
g = 10
l = 1
beta = 0.0  # Damping coefficient
theta0 = np.pi*0.1  # Initial angle

N = 1001
t = np.linspace(0, 20, N)  # Time array
theta = np.zeros(N)  # Angle array
theta[0] = theta0  # Set initial angle
omega = np.zeros(N)  # Angular velocity array
gamma = np.zeros(N)  # Angular acceleration array
for i in range(0, N-1):
    # Calculate acceleration
    gamma[i] = -(g/l) * np.sin(theta[i]) - beta * omega[i]  # Hooke's law
    omega[i+1] = omega[i] + gamma[i] * (t[i+1] - t[i])  # Angular velocity change over the time step

    # Update angle using simple Euler integration
    theta[i+1] = theta[i] + omega[i+1] * (t[i+1] - t[i])  # Update angle based on angular velocity

np.savetxt('pendulum_data.dat', np.column_stack((t, theta, omega, gamma)), header='# Time(s) Angle(rad) Angular Velocity(rad/s) Angular Acceleration(rad/s^2)', comments='')

bins = 1*N
signal_fft = np.fft.fft(theta, n = bins)
freq = np.fft.fftfreq(bins, d=(t[1]-t[0]))
mask = freq > 0
freq = freq[mask]
mag = 2.0/N * np.abs(signal_fft[mask])

np.savetxt('pendulum_fft.dat', np.column_stack((freq, mag)), header='# Frequency(Hz) Magnitude', comments='')

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(t, theta, label='Angle (theta)', color='blue', marker='o')
plt.title('Simple Harmonic Motion of a Pendulum')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(freq, mag, label='FFT Magnitude', color='red', marker='o')
plt.title('Frequency Spectrum of the Pendulum Motion')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 2)  # Limit x-axis to focus on low frequencies
plt.show()

# Animation of the pendulum motion
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1.2 * l, 1.2 * l)
ax.set_ylim(-1.2 * l, 1.2 * l)
ax.set_aspect('equal')
ax.set_title('Animated Pendulum Motion')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(False)
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

x = l * np.sin(theta)
y = -l * np.cos(theta)

line, = ax.plot([], [], color='white', lw=2, marker='o', markersize=8)


def init():
    line.set_data([], [])
    return (line,)


def animate(i):
    line.set_data([0, x[i]], [0, y[i]])
    return (line,)

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=N, interval=20, blit=False)
# ani.save('pendulum_animation.gif', writer='pillow', fps=30)
# print('Animation saved to pendulum_animation.gif')
plt.show()
