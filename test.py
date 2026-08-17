import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
    

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print(f"CPU: {torch.get_num_threads()} threads")

device = torch.device("cpu")  # Force CPU usage for debugging
print(f"Using device: {device} ({torch.get_num_threads()} threads)")

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
    'figure.figsize': (10/2.54, 6/2.54),  # 10x6 cm in inches (1 figure per line)
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


# Time formatting for runtime display
def time_format(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

data = np.loadtxt('pendulum_data.dat', skiprows=1)
t_num = data[:, 0]
theta_num = data[:, 1]
omega_num = data[:, 2]

# idx = np.array([0])
# idx = np.append(idx, np.random.choice(len(t_num), size=99, replace=False))
# idx = idx.flatten()

# t_data = t_num[idx]
# theta_data = theta_num[idx]
# omega_data = omega_num[idx]

t_data = data[:350:35, 0]
theta_data = data[:350:35, 1]
omega_data = data[:350:35, 2]


plt.plot(t_data, theta_data, label=r'Training Data', color='blue', marker='o', ls = 'None')
plt.plot(t_num, theta_num, label=r'Numerical Solution', color='orange', ls = '-')
plt.xlabel(r'Time (s)')
plt.ylabel(r'Angle (rad)')
plt.title(r'Pendulum PINN')
plt.legend()
plt.show()

# keep numpy copies for plotting/animation after we convert to tensors
t_data_np = t_data.copy()
theta_data_np = theta_data.copy()
omega_data_np = omega_data.copy()

t_data = torch.tensor(t_data, dtype=torch.float32, device=device).view(-1, 1)
theta_data = torch.tensor(theta_data, dtype=torch.float32, device=device).view(-1, 1)
omega_data = torch.tensor(omega_data, dtype=torch.float32, device=device).view(-1, 1)

t_min = t_num.min()
t_max = t_num.max()

#### PINN ####
##############

start_time = time.time()

class PINN(nn.Module):
    def __init__(self, alpha_init=0.0, beta_init=0.0):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

        # Learnable physical parameters (scalars)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

    def forward(self, t):
        t_scaled = 2 * (t - t_min) / (t_max - t_min) - 1
        return self.network(t_scaled)


alpha_init = 0.0
beta_init = 0.0

model = PINN(alpha_init=alpha_init, beta_init=beta_init).to(device)

t_physics = torch.linspace(t_min, t_max, 1000, dtype=torch.float32, device=device).view(-1, 1)
t_physics.requires_grad = True

# Prepare PINN prediction grid and interactive plot for live updates
t_PINN = np.linspace(t_min.item(), t_max.item(), 1000)
t_PINN_tensor = torch.tensor(t_PINN, dtype=torch.float32, device=device).view(-1, 1)


######### ANIMATION SECTION #############
#########################################
# plt.ion()
fig_anim, ax_anim = plt.subplots()
ax_anim.plot(t_num, theta_num, label=r'Numerical Solution', color='orange')
ax_anim.plot(t_data_np, theta_data_np, 'o', label=r'Training Data', color='blue')
pinn_line, = ax_anim.plot(t_PINN, np.zeros_like(t_PINN), label=r'PINN Prediction', color='red')
ax_anim.set_xlabel(r'Time (s)')
ax_anim.set_ylabel(r'Angle (rad)')
ax_anim.set_title(r'Pendulum PINN')
ax_anim.legend()
# fig_anim.show()

# collect PINN snapshots (one per update) for GIF saving
pinn_snapshots = []

### Optimization ###
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 1000000
lamb_data = 1e3
lamb_physics = 1e2
lamb_init = 5e1

# History for plotting
loss_history = []
data_loss_history = []
physics_loss_history = []
init_loss_history = []

for epoch in range(epochs+1):
    optimizer.zero_grad()
    theta_prediction = model(t_data)
    data_loss = torch.mean((theta_prediction - theta_data)**2)

    theta_physics = model(t_physics)
    omega_physics = torch.autograd.grad(
        theta_physics, t_physics, grad_outputs=torch.ones_like(theta_physics), create_graph=True
    )[0]  # Angular velocity
    gamma_physics = torch.autograd.grad(
        omega_physics, t_physics, grad_outputs=torch.ones_like(omega_physics), create_graph=True
    )[0]  # Angular acceleration


    # Use learnable scalar parameters from the model (broadcast automatically)
    residual_physics = gamma_physics + model.alpha * torch.sin(theta_physics) + model.beta * omega_physics
    physics_loss = torch.mean(residual_physics**2)

    residual_init = (theta_physics[0] - theta_data[0])**2 + (omega_physics[0] - omega_data[0])**2  # Initial condition residual
    init_loss = torch.mean(residual_init)

    loss = lamb_data * data_loss + lamb_physics * physics_loss + lamb_init * init_loss
    loss.backward()
    optimizer.step()

    # record
    loss_history.append(loss.item())
    data_loss_history.append(data_loss.item())
    physics_loss_history.append(physics_loss.item())
    init_loss_history.append(init_loss.item())

    # Print progress
    elapsed_time = time.time() - start_time
    if epoch % 100 == 0:
        print(f'\rEpoch {epoch}, Loss: {loss.item():.6f}, alpha: {model.alpha.item():.6f}, beta: {model.beta.item():.6f}, Time: {time_format(elapsed_time)}', end='', flush=True)


    #### ANIMATION ######
    # Update animated PINN prediction every 100 epochs
    if epoch % 1000 == 0:
        model.eval()
        with torch.no_grad():
            theta_PINN_now = model(t_PINN_tensor).cpu().numpy().flatten()
        # store snapshot for later animation saving
        pinn_snapshots.append(theta_PINN_now.copy())
        pinn_line.set_ydata(theta_PINN_now)
        ax_anim.set_title(f'Pendulum PINN - Epoch {epoch}')
        fig_anim.canvas.draw()
        fig_anim.canvas.flush_events()
        model.train()

# Learned parameters
alpha_learned = model.alpha.item()
beta_learned = model.beta.item()

print(f'Learned alpha: {alpha_learned:.6f}')
print(f'Learned beta: {beta_learned:.6f}')
print(f'Runtime: {time_format(elapsed_time)}')

# PINN Prediction
t_PINN = np.linspace(t_min.item(), t_max.item(), 1000)
t_PINN_tensor = torch.tensor(t_PINN, dtype=torch.float32, device=device).view(-1, 1)
model.eval()
with torch.no_grad():
    theta_PINN = model(t_PINN_tensor).cpu().numpy().flatten()


############### SAVE ANIMATION ######################
# Save collected snapshots as a GIF: 30 fps, each frame = 100 epochs
if len(pinn_snapshots) > 0:
    import matplotlib.animation as animation
    # plt.ioff()
    def update_frame(i):
        pinn_line.set_ydata(pinn_snapshots[i])
        ax_anim.set_title(f'Pendulum PINN — Epoch {i*1000}')
        return pinn_line,

    anim = animation.FuncAnimation(fig_anim, update_frame, frames=len(pinn_snapshots), blit=True)
    writer = animation.PillowWriter(fps=30)
    out_path = 'pinn_training.gif'
    anim.save(out_path, writer=writer)
    print(f'Saved animation to {out_path}')
    plt.close(fig_anim)

# Plotting the results
plt.plot(t_num, theta_num, label=r'Numerical Solution', color='orange', ls = '-')
plt.plot(t_data.cpu().numpy(), theta_data.cpu().numpy(), label=r'Training Data', color='blue', marker='o', ls = 'None')
plt.plot(t_PINN, theta_PINN, label=r'PINN Prediction', color='red')
plt.xlabel(r'Time (s)')
plt.ylabel(r'Angle (rad)')
plt.title(r'Pendulum PINN')
plt.legend()
plt.savefig("pinn_pendulum_results.png", dpi=600)
plt.show()


# # Plot loss histories
# epochs_arr = np.arange(len(loss_history))
# plt.semilogy(epochs_arr, loss_history, label='Total Loss', color='black')
# plt.semilogy(epochs_arr, data_loss_history, label='Data Loss', color='blue')
# plt.semilogy(epochs_arr, physics_loss_history, label='Physics Loss', color='red')
# plt.semilogy(epochs_arr, init_loss_history, label='Initial Condition Loss', color='green')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Convergence')
# plt.legend()
# plt.show()