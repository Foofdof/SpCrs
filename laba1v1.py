import numpy as np
import matplotlib.pyplot as plt

# -------------------- Problem-wide constants -------------------- #

gamma   = 1.4           # Cp/Cv, ideal gas
rho0val = 1.0           # reference density
h       = 0.001          # uniform spatial step (x-grid / mass grid)
t_max   = 0.1
CFL     = 0.4           # Courant number
Cq      = 1.0           # artificial viscosity coefficient (1–2 typical)

xgrid = np.arange(0.0, 1.0 + h, h)   # Eulerian grid (also initial Lagr. marks)

# ---------------------- Initial conditions ---------------------- #

def gaussian_pressure(x, p0=1.0, x0=0.5, sigma=0.05):
    """P(x,0) = P0·exp[ - ((x-x0)/sigma)^2 ]"""
    return p0 * np.exp(-((x - x0) / sigma) ** 2)

# -------------------- Boundary conditions ----------------------- #

def rigid_bc(u, rho, p):
    """Rigid wall: u = 0, dp/dx = 0"""
    u[0] = u[-1] = 0.0
    p[0], p[-1]     = p[1], p[-2]
    rho[0], rho[-1] = rho[1], rho[-2]

def free_bc(u, rho, p):
    """Free surface: p = 0, du/dx = 0"""
    p[0] = p[-1] = 0.0
    u[0], u[-1]     = u[1], u[-2]
    rho[0], rho[-1] = rho[1], rho[-2]

# ################################################################ #
#                       EULERian solvers                           #
# ################################################################ #

def _safe_dt(u, rho, p):
    """CFL time step for Eulerian schemes (max over grid)."""
    c = np.sqrt(gamma * p / rho)
    return CFL * h / np.max(np.abs(u) + c)

# ----------------  Lax, non-conservative  ----------------------- #

def lax_nc_step(u, rho, p, dt, dx):
    rho_new, u_new, p_new = rho.copy(), u.copy(), p.copy()

    rho_new[1:-1] = (
        0.5 * (rho[2:] + rho[:-2])
        - u[1:-1] * dt / (2 * dx) * (rho[2:] - rho[:-2])
        - rho[1:-1] * dt / (2 * dx) * (u[2:] - u[:-2])
    )
    u_new[1:-1] = (
        0.5 * (u[2:] + u[:-2])
        - u[1:-1] * dt / (2 * dx) * (u[2:] - u[:-2])
        - dt / (2 * dx * rho[1:-1]) * (p[2:] - p[:-2])
    )
    p_new[1:-1] = (
        0.5 * (p[2:] + p[:-2])
        - u[1:-1] * dt / (2 * dx) * (p[2:] - p[:-2])
        - gamma * p[1:-1] * dt / (2 * dx) * (u[2:] - u[:-2])
    )
    return u_new, rho_new, p_new

# ----------------  Euler up-wind (flow-directed) ---------------- #

def euler_upwind_step(u, rho, p, dt, dx):
    rho_new, u_new, p_new = rho.copy(), u.copy(), p.copy()

    alpha_full = np.where(u > 0, 0.0, 1.0)
    a = alpha_full[1:-1]

    rho_new[1:-1] = (
        rho[1:-1]
        - (1 - a) * u[1:-1] * dt / dx * (rho[1:-1] - rho[:-2])
        - a * u[1:-1] * dt / dx * (rho[2:] - rho[1:-1])
        - rho[1:-1] * dt / (2 * dx) * (u[2:] - u[:-2])
    )
    u_new[1:-1] = (
        u[1:-1]
        - (1 - a) * u[1:-1] * dt / dx * (u[1:-1] - u[:-2])
        - a * u[1:-1] * dt / dx * (u[2:] - u[1:-1])
        - dt / (2 * dx * rho[1:-1]) * (p[2:] - p[:-2])
    )
    p_new[1:-1] = (
        p[1:-1]
        - (1 - a) * u[1:-1] * dt / dx * (p[1:-1] - p[:-2])
        - a * u[1:-1] * dt / dx * (p[2:] - p[1:-1])
        - gamma * p[1:-1] * dt / (2 * dx) * (u[2:] - u[:-2])
    )
    return u_new, rho_new, p_new

# ----------------  Lax, conservative  --------------------------- #

def lax_c_step(u, rho, p, dt, dx):
    m    = rho * u
    Eint = p / (gamma - 1)
    Etot = Eint + 0.5 * rho * u**2

    F_r = m
    F_m = m * u + p
    F_E = (Etot + p) * u

    rho_new = rho.copy()
    m_new   = m.copy()
    E_new   = Etot.copy()

    rho_new[1:-1] = 0.5 * (rho[2:] + rho[:-2]) - dt /(2*dx) * (F_r[2:] - F_r[:-2])
    m_new  [1:-1] = 0.5 * (m  [2:] + m  [:-2]) - dt /(2*dx) * (F_m[2:] - F_m[:-2])
    E_new  [1:-1] = 0.5 * (Etot[2:] + Etot[:-2]) - dt /(2*dx) * (F_E[2:] - F_E[:-2])

    rho_new = np.maximum(rho_new, 1e-16)
    u_new   = m_new / rho_new
    p_new   = (gamma - 1) * (E_new - 0.5 * rho_new * u_new**2)
    return u_new, rho_new, p_new

# ----------------  Euler driver (common)  ------------------------ #

def euler_calc(stepper, bc_func, title):
    rho = np.full_like(xgrid, rho0val)
    u   = np.zeros_like(xgrid)
    p   = gaussian_pressure(xgrid)
    e = p / (gamma - 1) + 0.5 * rho * u ** 2

    t_hist, rho_hist, u_hist, p_hist, e_hist = [0.0], [rho.copy()], [u.copy()], [p.copy()], [e.copy()]

    t = 0.0
    while t < t_max:
        dt = _safe_dt(u, rho, p)
        if t + dt > t_max:
            dt = t_max - t

        u, rho, p = stepper(u, rho, p, dt, h)
        bc_func(u, rho, p)

        e = p / (gamma - 1) + 0.5 * rho * u ** 2

        t += dt
        t_hist.append(t)
        rho_hist.append(rho.copy())
        u_hist.append(u.copy())
        p_hist.append(p.copy())
        e_hist.append(e.copy())

    _plot_fields(xgrid, t_hist, [p_hist, u_hist, rho_hist, e_hist], title)

# ################################################################ #
#            Lagrangian Richtmyer-Morton + artificial Q           #
# ################################################################ #

def artificial_viscosity(u):
    """
    Von Neumann–Richtmyer artificial viscosity.
    Для каждого ребра j+1/2 вычисляем:
      δ = u[j+1] - u[j]
      q_edge = Cq * rho0 * δ^2  если δ < 0, иначе 0
    А затем узловую q[j] = ½ (q_edge[j-1] + q_edge[j]).
    """
    # разности скоростей на «рёбрах»
    delta_u = u[1:] - u[:-1]
    q_edge = np.zeros_like(delta_u)
    # только при сжатии (δ<0)
    mask = delta_u < 0.0
    q_edge[mask] = Cq * rho0val * delta_u[mask]**2
    # интерполируем в узлы
    q = np.zeros_like(u)
    q[1:-1] = 0.5 * (q_edge[:-1] + q_edge[1:])
    return q

def rm_lagrange_step(u, R, V, p, E, dt, alpha):
    """
    Predictor–corrector схема Рихтмайера–Мортона с искусственной вязкостью.

    1) Вычисляем q = artificial_viscosity(u)
    2) В полушаге используем p ± q:
         Vh = ½ (V[j] + V[j+1]) + (dt/2)/Δξ * (u[j+1] - u[j])
         uh = ½ (u[j] + u[j+1]) - (dt/2)/Δξ * ((p+q)[j+1] - (p+q)[j])
         Eh = ½ (E[j] + E[j+1])
         ph = (γ-1)Eh/Vh
    3) В корректоре обновляем V, u, E в узлах:
         V_new[j] = V[j] + dt * (uh[j] - uh[j-1]) / Δξ[j]
         u_new[j] = u[j] - dt * ( (ph+qh)[j] - (ph+qh)[j-1] ) / Δξ[j]
         E_new[j] = E[j] - dt * ( (ph+qh)[j]*uh[j] - (ph+qh)[j-1]*uh[j-1] ) / Δξ[j]
    4) Обновляем R: R[j] += dt * u_new[j]
    """
    # искусственная вязкость на узлах
    q = artificial_viscosity(u)
    # вводим «ξ»-координату для α-симметрии
    xi = R if alpha == 1 else R**(alpha-1)/(alpha-1)
    dxi = np.diff(xi)

    # значения слева/справа на рёбрах
    uL, uR = u[:-1], u[1:]
    VL, VR = V[:-1], V[1:]
    pL = p[:-1] + q[:-1]
    pR = p[1:]  + q[1:]

    # predictor: полушаг
    Vh = 0.5*(VL + VR) + 0.5 * dt/dxi * (uR - uL)
    uh = 0.5*(uL + uR) - 0.5 * dt/dxi * (pR - pL)
    Eh = 0.5*(E[:-1] + E[1:])
    ph = (gamma - 1) * Eh / Vh
    qh = 0.5*(q[:-1] + q[1:])

    # corrector: полный шаг
    Vn = V.copy()
    un = u.copy()
    En = E.copy()

    # обновляем только внутренние узлы
    Vn[1:-1] += dt * (uh[1:]   - uh[:-1])   / dxi[1:]
    un[1:-1] += -dt * ((ph+qh)[1:] - (ph+qh)[:-1]) / (dxi[1:] * 1.0)
    En[1:-1] += -dt * ((ph+qh)[1:]*uh[1:] - (ph+qh)[:-1]*uh[:-1]) / dxi[1:]

    # давление по новому E и V
    pn = (gamma - 1) * En / Vn

    # двигаем маркеры
    R[1:-1] += dt * un[1:-1]

    return un, Vn, pn, En

def lagrange_calc(bc_func, alpha=1):
    R = xgrid.copy()

    V = np.full_like(xgrid, 1.0 / rho0val)
    p = gaussian_pressure(xgrid)
    E = p * V / (gamma - 1)
    u = np.zeros_like(xgrid)

    t_hist, rho_hist, u_hist, p_hist, E_hist = [0.0], [1.0/V], [u.copy()], [p.copy()], [E.copy()]

    t = 0.0
    while t < t_max:
        c = np.sqrt(gamma * p * V)
        dt = CFL * h / np.max(np.abs(u) + c)
        if t + dt > t_max:
            dt = t_max - t
        # dt = 0.5 * h / (max(np.abs(u)) + np.sqrt(gamma * min(np.abs(p)) / max(np.abs(1/V))))
        dt = min(dt, 0.001)

        u, V, p, E = rm_lagrange_step(u, R, V, p, E, dt, alpha)
        rho = 1.0 / V

        bc_func(u, rho, p)
        V[:] = 1.0 / rho
        E[:] = p * V / (gamma - 1)

        t += dt
        t_hist.append(t)
        rho_hist.append(rho.copy())
        u_hist.append(u.copy())
        p_hist.append(p.copy())
        E_hist.append(E.copy())

    geom = {1:"Plane", 2:"Cyl.", 3:"Sph."}[alpha]
    _plot_fields(xgrid, t_hist, [p_hist, u_hist, rho_hist, E_hist], f"Lagrange {geom}")


def _plot_fields(x, t_hist, data_list, title):
    idx = np.linspace(0, len(t_hist)-1, 5, dtype=int)
    lbl = ["Pressure P", "Velocity u", "Density ρ"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for ax, d, l in zip(axes, data_list[:3], lbl):
        for k, col in zip(idx, plt.cm.viridis(np.linspace(0,1,len(idx)))):
            ax.plot(x, d[k], color=col, label=f"t={t_hist[k]:.3f}")
        ax.set_ylabel(l)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x (or R)")
    fig.suptitle(title)
    plt.tight_layout(); plt.show()

    total_E = [np.sum(data_list[3][i]) * h for i in idx]
    plt.figure(figsize=(6, 4))
    plt.plot([t_hist[i] for i in idx], total_E, 'o-')
    plt.xlabel('t')
    plt.ylabel('∫E dx')
    plt.title('Total energy vs time')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Euler
    euler_calc(lax_nc_step, rigid_bc, "Euler: Lax non-cons. Rigid-BC")
    euler_calc(euler_upwind_step, rigid_bc, "Euler: Up-wind. Rigid-BC")
    euler_calc(lax_c_step,  rigid_bc, "Euler: Lax conservative. Rigid-BC")

    euler_calc(lax_nc_step, free_bc, "Euler: Lax non-cons. Free-BC")
    euler_calc(euler_upwind_step, free_bc, "Euler: Up-wind. Free-BC")
    euler_calc(lax_c_step, free_bc, "Euler: Lax conservative. Free-BC")

    # Lagrange
    for a in (1, 2, 3):
        lagrange_calc(rigid_bc, alpha=a)

    for a in (1, 2, 3):
        lagrange_calc(free_bc, alpha=a)
