import numpy as np
import matplotlib.pyplot as plt

# domain geometry
domain_length_x, domain_length_y = 2.0, 1.0
num_cells_x, num_cells_y = 161, 81
delta_x = domain_length_x / (num_cells_x - 1)
delta_y = domain_length_y / (num_cells_y - 1)

# fluid and scalar properties
rho_fluid = 1.0
nu_fluid = 0.5
u_inlet = 5.0
diffusivity_scalar = 0.1

# time integration
delta_t = 2e-05
total_steps = 500_000
pressure_poisson_iters = 5

# rectangular obstacle indices
obst_x0, obst_y0 = 0.25, 0.25
obst_width, obst_height = 0.2, 0.5
obst_ix0 = int(obst_x0 / delta_x)
obst_ix1 = int((obst_x0 + obst_width) / delta_x)
obst_iy0 = int(obst_y0 / delta_y)
obst_iy1 = int((obst_y0 + obst_height) / delta_y)

num_snapshots = 10
snapshot_interval = total_steps // num_snapshots
snapshot_count = 0

def apply_velocity_bc(u, v):
    u[0, :] = u_inlet
    v[0, :] = 0.0
    u[-1, :] = u[-2, :]
    v[-1, :] = v[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]
    u[obst_ix0:obst_ix1+1, obst_iy0:obst_iy1+1] = 0.0
    v[obst_ix0:obst_ix1+1, obst_iy0:obst_iy1+1] = 0.0

def apply_scalar_bc(scalar):
    scalar[0, :] = 1.0
    scalar[-1, :] = scalar[-2, :]
    scalar[:, 0] = scalar[:, 1]
    scalar[:, -1] = scalar[:, -2]
    scalar[obst_ix0:obst_ix1+1, obst_iy0:obst_iy1+1] = 0.0

def save_snapshot(u, v, scalar, step):
    global snapshot_count
    speed = np.sqrt(u**2 + v**2)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(
        speed.T, origin='lower', cmap='jet',
        extent=[0, domain_length_x, 0, domain_length_y]
    )
    axes[0].set_title(f"Speed, t={delta_t*step:.6f}s")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(
        scalar.T, origin='lower', cmap='jet',
        extent=[0, domain_length_x, 0, domain_length_y]
    )
    axes[1].set_title(f"Scalar, t={delta_t*step:.6f}s")
    fig.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"snapshot_{snapshot_count+1}.png")
    plt.close(fig)
    snapshot_count += 1
    print(f"Saved snapshot_{snapshot_count}.png")

def main():
    u = np.zeros((num_cells_x, num_cells_y), dtype=np.float32)
    v = np.zeros((num_cells_x, num_cells_y), dtype=np.float32)
    p = np.zeros((num_cells_x, num_cells_y), dtype=np.float32)
    b = np.zeros((num_cells_x, num_cells_y), dtype=np.float32)
    u_star = np.zeros_like(u)
    v_star = np.zeros_like(v)
    scalar_field = np.zeros((num_cells_x, num_cells_y), dtype=np.float32)

    apply_scalar_bc(scalar_field)

    for step in range(1, total_steps + 1):
        u_star[1:-1,1:-1] = (
            u[1:-1,1:-1]
            + delta_t * (
                - (u[2:,1:-1]**2 - u[:-2,1:-1]**2) / (2*delta_x)
                - (u[1:-1,2:]*v[1:-1,2:] - u[1:-1,:-2]*v[1:-1,:-2]) / (2*delta_y)
                + nu_fluid * (
                    (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / delta_x**2
                    + (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / delta_y**2
                )
            )
        )
        v_star[1:-1,1:-1] = (
            v[1:-1,1:-1]
            + delta_t * (
                - (u[2:,1:-1]*v[2:,1:-1] - u[:-2,1:-1]*v[:-2,1:-1]) / (2*delta_x)
                - (v[1:-1,2:]**2 - v[1:-1,:-2]**2) / (2*delta_y)
                + nu_fluid * (
                    (v[2:,1:-1] - 2*v[1:-1,1:-1] + v[:-2,1:-1]) / delta_x**2
                    + (v[1:-1,2:] - 2*v[1:-1,1:-1] + v[1:-1,:-2]) / delta_y**2
                )
            )
        )
        apply_velocity_bc(u_star, v_star)

        b[1:-1,1:-1] = (
            (rho_fluid/delta_t) * (
                (u_star[2:,1:-1] - u_star[:-2,1:-1]) / (2*delta_x)
                + (v_star[1:-1,2:] - v_star[1:-1,:-2]) / (2*delta_y)
            )
        )

        for _ in range(pressure_poisson_iters):
            p[1:-1,1:-1] = (
                ((p[2:,1:-1] + p[:-2,1:-1]) * delta_y**2
               + (p[1:-1,2:] + p[1:-1,:-2]) * delta_x**2
               - b[1:-1,1:-1] * delta_x**2 * delta_y**2)
                / (2*(delta_x**2 + delta_y**2))
            )
            p[:,0] = p[:,1]
            p[:,-1] = p[:,-2]
            p[0,:] = p[1,:]
            p[-1,:] = p[-2,:]

        u[1:-1,1:-1] = (
            u_star[1:-1,1:-1]
            - delta_t * (p[2:,1:-1] - p[:-2,1:-1]) / (2*delta_x)
        )
        v[1:-1,1:-1] = (
            v_star[1:-1,1:-1]
            - delta_t * (p[1:-1,2:] - p[1:-1,:-2]) / (2*delta_y)
        )
        apply_velocity_bc(u, v)

        old_scalar = scalar_field.copy()
        scalar_field[1:-1,1:-1] = (
            old_scalar[1:-1,1:-1]
            + delta_t * (
                - u[1:-1,1:-1] * (old_scalar[2:,1:-1] - old_scalar[:-2,1:-1]) / (2*delta_x)
                - v[1:-1,1:-1] * (old_scalar[1:-1,2:] - old_scalar[1:-1,:-2]) / (2*delta_y)
                + diffusivity_scalar * (
                    (old_scalar[2:,1:-1] - 2*old_scalar[1:-1,1:-1] + old_scalar[:-2,1:-1]) / delta_x**2
                    + (old_scalar[1:-1,2:] - 2*old_scalar[1:-1,1:-1] + old_scalar[1:-1,:-2]) / delta_y**2
                )
            )
        )
        apply_scalar_bc(scalar_field)

        if step % snapshot_interval == 0 and snapshot_count < num_snapshots:
            save_snapshot(u, v, scalar_field, step)

    print(f"Simulation complete, {snapshot_count} snapshots saved.")

if __name__ == '__main__':
    main()