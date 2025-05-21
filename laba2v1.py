import numpy as np
import matplotlib.pyplot as plt

# inlet velocity
u0 = 4.

# Reynolds
Re = 7000.0

# time discretization
t_max = 0.3
nt = 3000
dt = t_max / nt

# size of region
Lx, Ly = 2., 2.
nx, ny = 20, 20
dx, dy = Lx / nx, Ly / ny

# cords of bottom left and top right corner of rect body:
# [ (x1,y1), (x2,y2) ]
bodies = [
    [(0.7, 0.5), (0.8, 1.5)],
]

x = np.linspace(0, Lx, nx+1)
y = np.linspace(0, Ly, ny+1)

X, Y = np.meshgrid(x, y, indexing='xy')

# Mask (fluid=True, solid=False)
fluid = np.ones_like(X, dtype=bool)
for (x1, y1), (x2, y2) in bodies:
    inside = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    fluid[inside] = False

# fields: curl=ksi, psi
psi = np.zeros_like(X, dtype=np.float64)
ksi = np.zeros_like(X, dtype=np.float64)

# pre-body layers
up   = np.vstack([fluid[0:1,:], fluid[:-1,:]])
down = np.vstack([fluid[1:,:],  fluid[-1:,:]])
left  = np.hstack([fluid[:,:1], fluid[:,:-1]])
right = np.hstack([fluid[:,1:],  fluid[:,-1:]])

up_left = np.roll(up, +1, axis=1)
up_right = np.roll(up, -1, axis=1)
down_left = np.roll(down, +1, axis=1)
down_right = np.roll(down, -1, axis=1)

pre_body_lrs = fluid & (~up | ~down | ~left | ~right)
pre_body_lrs2 = (
    fluid & (
        ~up | ~down
        | ~left | ~right
        | ~up_left | ~up_right
        | ~down_left | ~down_right
    )
)

#inlet, outlet masks
inlet_wall = np.zeros_like(X, dtype=bool)
inlet_wall[(X == x[0])] = True
outlet_wall = np.zeros_like(X, dtype=bool)
outlet_wall[(X == x[-1])] = True

# top, bottom boundary
top_wall = np.zeros_like(X, dtype=bool)
top_wall[(Y == y[-1])] = True
bottom_wall = np.zeros_like(X, dtype=bool)
bottom_wall[(Y == y[0])] = True


# boundary conditions for ksi
def apply_ksi_bc():
    # ksi[0, :] = -2 * (psi[1, :] - psi[0, :] + u0 * dy) / dy ** 2
    # ksi[-1, :] = 2 * (psi[-2, :] - psi[-1, :] + u0 * dy) / dy ** 2

    ksi[0,  :] = ksi[-2, :]
    ksi[-1, :] = ksi[1,  :]

    # вход/выход – нулевой градиент вдоль x
    ksi[:, 0] = ksi[:, 1]
    ksi[:, -1] = ksi[:, -2]

    for i, j in zip(*np.where(pre_body_lrs2)):
        left_w = pre_body_lrs2[i, j+1]
        right_w = pre_body_lrs2[i, j-1]
        up_w = pre_body_lrs2[i+1, j]
        down_w = pre_body_lrs2[i-1, j]

        left_b = not fluid[i, j + 1]
        right_b = not fluid[i, j - 1]
        up_b = not fluid[i + 1, j]
        down_b = not fluid[i - 1, j]

        # corners
        # ld
        if right_w and up_w:
            ksi[i, j] = (2 * psi[i, j - 1] / dx ** 2 + 2 * psi[i - 1, j] / dy ** 2)
            continue
        # rd
        if left_w and up_w:
            ksi[i, j] = (2 * psi[i, j + 1] / dx ** 2 + 2 * psi[i - 1, j] / dy ** 2)
            continue
        # ru
        if left_w and down_w:
            ksi[i, j] = 2 * psi[i, j + 1] / dx ** 2 + 2 * psi[i + 1, j] / dy ** 2
            continue
        # lu
        if right_w and down_w:
            ksi[i, j] = 2 * psi[i, j - 1] / dx ** 2 + 2 * psi[i + 1, j] / dy ** 2
            continue

        # without corners
        # horizontal walls
        if left_w and right_w:
            if down_b:
                ksi[i, j] = -2 * psi[i + 1, j] / dy ** 2
            else:
                ksi[i, j] = -2 * psi[i - 1, j] / dy ** 2
            continue
        # vertical walls
        if up_w and down_w:
            if left_b:
                ksi[i, j] = -2 * psi[i, j + 1] / dx ** 2
            else:
                ksi[i, j] = -2 * psi[i, j - 1] / dx ** 2
            continue

    return ksi

# boundary conditions for psi
def apply_psi_bc():
    psi[pre_body_lrs] = 0.0
    psi[0, :] = psi[1, :] - u0 * dy
    psi[-1, :] = psi[-2, :] + u0 * dy
    # psi[0, :] = psi[1, :]
    # psi[-1, :] = psi[-2, :]

    psi[:, 0] = psi[:, 1]                   # inlet grad = 0
    psi[:, -1] = psi[:, -2]                 # outlet grad = 0

# def apply_psi_bc():
#     psi[pre_body_lrs] = 0.0
#     psi[:, 0] = u0 * y                      # inlet
#     psi[:, 1] = psi[:, 0]                   # inlet grad = 0
#     psi[:, -1] = psi[:, -2]                 # outlet grad = 0
#
#     psi[0, :] = 0
#     psi[-1, :] = 0
#
#     # psi[0, :] = psi[1, :] - u0 * dy         # bottom wall
#     # psi[-1, :] = psi[-2, :] + u0 * dy       # top wall

# poisson equation solver

def solve_poisson():
    ax, ay = dx ** 2, dy ** 2
    den = 2 * (ax + ay)

    max_iter = 50000
    eps = 1e-4
    omega = 1.7
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny):
            for j in range(1, nx):
                if (fluid[i, j]
                        and not inlet_wall[i, j]
                        and not outlet_wall[i, j]
                        and not bottom_wall[i, j]
                        and not top_wall[i, j]
                        and not pre_body_lrs2[i, j]):
                    val = (
                        (psi[i + 1, j] + psi[i - 1, j]) * ax / den +
                        (psi[i, j + 1] + psi[i, j - 1]) * ay / den +
                        ksi[i, j] * ax * ay / den
                    )
                    new = omega * val + (1 - omega) * psi[i, j]
                    diff = abs(new - psi[i, j])
                    if diff > max_diff:
                        max_diff = diff
                    psi[i, j] = new

        apply_psi_bc()
        if max_diff < eps:
            # print(f"Poisson solver converged in {_} iter; err={max_diff:.2e}")
            break
    else:
        print(f"Poisson solver reached max_iter={max_iter}; last err={max_diff:.2e}")

    return psi

# def solve_poisson():
#     ax, ay = dx**2, dy**2
#     den    = 2*(ax + ay)
#
#     max_iter = 15000
#     eps      = 1e-4
#     omega    = 1.7
#
#     # заранее собираем все узлы, где надо обновлять ψ
#     interior_mask = (
#         fluid
#         & ~inlet_wall
#         & ~outlet_wall
#         & ~bottom_wall
#         & ~top_wall
#         & ~pre_body_lrs2
#     )
#     pts = list(zip(*np.where(interior_mask)))  # [(i1,j1), (i2,j2), ...]
#
#     for it in range(1, max_iter+1):
#         max_diff = 0.0
#
#         # один цикл по всем interior-пунктам
#         for i, j in pts:
#             # счётчик обхода (i,j) тот же, что и раньше
#             val = (
#                 (psi[i+1, j] + psi[i-1, j]) * ax / den
#               + (psi[i, j+1] + psi[i, j-1]) * ay / den
#               + ksi[i, j] * ax * ay / den
#             )
#             new = omega * val + (1 - omega) * psi[i, j]
#             diff = abs(new - psi[i, j])
#             if diff > max_diff:
#                 max_diff = diff
#             psi[i, j] = new
#
#         apply_psi_bc()
#
#         if max_diff < eps:
#             print(f"Poisson solver converged in {it} iter; err={max_diff:.2e}")
#             break
#     else:
#         print(f"Poisson solver reached max_iter={max_iter}; last err={max_diff:.2e}")
#
#     return psi

def ksi_step(u, v):
    ksi_new = ksi.copy()

    alpha = np.where(u > 0, 0, 1)[1:-1, 1:-1]
    beta = np.where(v > 0, 0, 1)[1:-1, 1:-1]

    ksi_c = ksi[1:-1, 1:-1]
    ksi_jl = ksi[:-2, 1:-1]
    ksi_jr = ksi[2:, 1:-1]
    ksi_il = ksi[1:-1, :-2]
    ksi_ir = ksi[1:-1, 2:]

    u_c = u[1:-1, 1:-1]
    v_c = v[1:-1, 1:-1]

    update = (
        ksi_c
        - dt/dx * u_c * (1 - alpha) * (ksi_c - ksi_il)
        - dt/dx * u_c * alpha * (ksi_ir - ksi_c)
        - dt / dy * v_c * (1 - beta) * (ksi_c - ksi_jl)
        - dt / dy * v_c * beta * (ksi_jr - ksi_c)
        + dt/Re * ((ksi_ir - 2*ksi_c + ksi_il)/dx**2 + (ksi_jr - 2*ksi_c + ksi_jl)/dy**2)
    )

    ksi_new[1:-1, 1:-1][fluid[1:-1, 1:-1]] = update[fluid[1:-1, 1:-1]]

    return ksi_new


from tqdm import trange

sample_steps = np.arange(0, nt, 500)
speed_profiles = []
time_points = []

u_full = np.zeros_like(psi)
v_full = np.zeros_like(psi)
u_full[:, 0] = u0

apply_psi_bc()
apply_ksi_bc()

for n in trange(nt, desc="Time stepping"):
    psi[:, 0] = u0 * y
    ksi = ksi_step(u_full, v_full)
    psi = solve_poisson()
    psi[:, 0] = u0 * y

    u = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dy)
    v = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)

    u_full[1:-1, 1:-1] = u
    v_full[1:-1, 1:-1] = v

    u_full[1:-1, -1] = u_full[1:-1, -2]
    v_full[1:-1, -1] = v_full[1:-1, -2]

    u_full[:, 0] = u0
    u_full[:, 1] = u0
    v_full[:, 0] = 0.0
    v_full[:, 1] = 0.0

    v_full[0, :] = 0.0
    v_full[-1, :] = 0.0
    u_full[0, :] = u_full[1, :]
    u_full[-1, :] = u_full[-2, :]

    ksi = apply_ksi_bc()

    if n in sample_steps:
        speed = np.sqrt(u_full ** 2 + v_full ** 2)
        speed_profiles.append(speed)
        time_points.append((n + 1) * dt)

        plt.figure(figsize=(12, 9))
        cf = plt.contourf(X, Y, speed, levels=50, cmap='viridis')
        plt.colorbar(cf, label="|V|")

        strm = plt.streamplot(
            X, Y, u_full, v_full,
            color='black', linewidth=1,
            density=1.5,  # регулирует густоту линий
            arrowsize=1.0
        )

        plt.title(f"Плотность |V| и линии тока, t = {dt * (n + 1):.2f} c")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()
