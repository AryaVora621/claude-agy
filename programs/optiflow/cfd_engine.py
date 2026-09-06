"""
OptiFlow 2D: Computational Fluid Dynamics (CFD) Engine
Zero external dependencies. Pure Python standard library.

Solves 2D Incompressible Navier-Stokes equations via Vorticity-Streamfunction
formulation with Poisson pressure recovery and aerodynamic force integration:

    Vorticity Transport:
        d(omega)/dt + u * d(omega)/dx + v * d(omega)/dy = nu * Laplacian(omega)

    Streamfunction Poisson Equation:
        Laplacian(psi) = -omega
        u = d(psi)/dy,  v = -d(psi)/dx

    Pressure Poisson Equation:
        Laplacian(p) = 2 * rho * (du/dx * dv/dy - du/dy * dv/dx)

    Aerodynamic Force Coefficients:
        C_L = 2 * F_lift / (rho * U_inf^2 * c)
        C_D = 2 * F_drag / (rho * U_inf^2 * c)
        L/D = C_L / C_D
"""

import math
from typing import List, Tuple, Dict, Optional


class Particle:
    """Smoke tracer particle for streakline flow visualization."""
    __slots__ = ('x', 'y', 'age', 'max_age')

    def __init__(self, x: float, y: float, max_age: float = 12.0):
        self.x = x
        self.y = y
        self.age = 0.0
        self.max_age = max_age

    def is_alive(self) -> bool:
        return self.age < self.max_age


class CFDEngine2D:
    """
    2D Incompressible Navier-Stokes finite difference solver
    with body-fitted obstacle masking and aerodynamic force evaluation.
    """

    def __init__(self, nx: int = 80, ny: int = 40, lx: float = 2.0, ly: float = 1.0,
                 u_inf: float = 1.0, reynolds: float = 200.0, rho: float = 1.225):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)

        self.u_inf = u_inf
        self.reynolds = reynolds
        self.rho = rho
        self.nu = (u_inf * ly) / max(1.0, reynolds)

        # 2D Grid Fields (flattened or nested lists)
        self.psi = [[0.0 for _ in range(nx)] for _ in range(ny)]
        self.omega = [[0.0 for _ in range(nx)] for _ in range(ny)]
        self.u = [[u_inf for _ in range(nx)] for _ in range(ny)]
        self.v = [[0.0 for _ in range(nx)] for _ in range(ny)]
        self.pressure = [[0.0 for _ in range(nx)] for _ in range(ny)]
        self.solid = [[False for _ in range(nx)] for _ in range(ny)]

        # Solid boundary cell tracking for force integration
        self.boundary_cells: List[Tuple[int, int, float, float]] = [] # (x, y, nx, ny)

        # Particles for flow visualization
        self.particles: List[Particle] = []
        self.particle_spawn_timer = 0.0

        # Telemetry metrics
        self.time = 0.0
        self.cl = 0.0
        self.cd = 0.0
        self.cm = 0.0
        self.lift_force = 0.0
        self.drag_force = 0.0
        self.max_velocity = u_inf
        self.chord_length = 0.5 * lx

        # Initialize uniform flow streamfunction
        self.reset_flow()

    def reset_flow(self):
        """Reset fields to uniform freestream condition."""
        self.time = 0.0
        for j in range(self.ny):
            y = j * self.dy
            for i in range(self.nx):
                self.psi[j][i] = self.u_inf * y
                self.omega[j][i] = 0.0
                self.u[j][i] = self.u_inf
                self.v[j][i] = 0.0
                self.pressure[j][i] = 0.0
        self.particles.clear()

    def clear_geometry(self):
        """Remove all solid obstacles from the domain."""
        for j in range(self.ny):
            for i in range(self.nx):
                self.solid[j][i] = False
        self.boundary_cells.clear()

    def update_boundary_normals(self):
        """Compute normal vectors on boundary cells for aerodynamic force integration."""
        self.boundary_cells.clear()
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid[j][i]:
                    # Check if adjacent to fluid
                    nbrs = [
                        (i + 1, j, 1.0, 0.0),
                        (i - 1, j, -1.0, 0.0),
                        (i, j + 1, 0.0, 1.0),
                        (i, j - 1, 0.0, -1.0)
                    ]
                    for ni, nj, norm_x, norm_y in nbrs:
                        if not self.solid[nj][ni]:
                            self.boundary_cells.append((i, j, norm_x, norm_y))

    def set_naca_airfoil(self, m: float = 0.0, p: float = 0.0, t: float = 0.12,
                          chord: float = 0.6, x_center: float = 0.7, y_center: float = 0.5,
                          alpha_deg: float = 5.0):
        """
        Generate NACA 4-digit airfoil profile at specified Angle of Attack.
        m: maximum camber (e.g. 0.02 for NACA 2412)
        p: location of maximum camber (e.g. 0.4 for NACA 2412)
        t: maximum thickness (e.g. 0.12 for 12%)
        alpha_deg: Angle of attack in degrees
        """
        self.clear_geometry()
        self.chord_length = chord
        alpha_rad = math.radians(alpha_deg)
        cos_a = math.cos(alpha_rad)
        sin_a = math.sin(alpha_rad)

        for j in range(self.ny):
            y_world = j * self.dy
            for i in range(self.nx):
                x_world = i * self.dx

                # Transform world coordinate to airfoil chord coordinate system
                dx = x_world - x_center
                dy = y_world - y_center

                # Rotate by -alpha to align with chord
                x_local = dx * cos_a + dy * sin_a + 0.5 * chord
                y_local = -dx * sin_a + dy * cos_a

                # Normalized chord station x/c in [0, 1]
                xc = x_local / chord
                if 0.0 <= xc <= 1.0:
                    # Half-thickness distribution
                    yt = 5.0 * t * chord * (
                        0.2969 * math.sqrt(xc) -
                        0.1260 * xc -
                        0.3516 * (xc ** 2) +
                        0.2843 * (xc ** 3) -
                        0.1015 * (xc ** 4)
                    )

                    # Camber line yc and camber slope
                    if p > 0.0 and m > 0.0:
                        if xc <= p:
                            yc = (m * chord / (p ** 2)) * (2.0 * p * xc - xc ** 2)
                        else:
                            yc = (m * chord / ((1.0 - p) ** 2)) * ((1.0 - 2.0 * p) + 2.0 * p * xc - xc ** 2)
                    else:
                        yc = 0.0

                    # Check if point lies inside upper and lower bounds
                    if abs(y_local - yc) <= yt:
                        self.solid[j][i] = True

        self.update_boundary_normals()

    def set_circular_cylinder(self, radius: float = 0.08, x_center: float = 0.6, y_center: float = 0.5):
        """Generate circular cylinder obstacle for Karman vortex shedding."""
        self.clear_geometry()
        self.chord_length = 2.0 * radius

        for j in range(self.ny):
            y_world = j * self.dy
            for i in range(self.nx):
                x_world = i * self.dx
                dist = math.hypot(x_world - x_center, y_world - y_center)
                if dist <= radius:
                    self.solid[j][i] = True

        self.update_boundary_normals()

    def set_venturi_nozzle(self, throat_height: float = 0.35, inlet_height: float = 0.75):
        """Generate convergent-divergent Venturi constriction."""
        self.clear_geometry()
        self.chord_length = self.lx * 0.5

        for j in range(self.ny):
            y_world = j * self.dy
            for i in range(self.nx):
                x_world = i * self.dx
                # Cosine constriction profile
                s = math.sin(math.pi * x_world / self.lx)
                wall_y = inlet_height - (inlet_height - throat_height) * s

                # Top and bottom symmetric constriction walls
                top_limit = 0.5 * self.ly + 0.5 * wall_y
                bot_limit = 0.5 * self.ly - 0.5 * wall_y

                if y_world >= top_limit or y_world <= bot_limit:
                    self.solid[j][i] = True

        self.update_boundary_normals()

    def set_backward_facing_step(self, step_x: float = 0.5, step_h: float = 0.3):
        """Generate backward-facing step creating recirculation bubble."""
        self.clear_geometry()
        self.chord_length = step_h

        for j in range(self.ny):
            y_world = j * self.dy
            for i in range(self.nx):
                x_world = i * self.dx
                if x_world <= step_x and y_world <= step_h:
                    self.solid[j][i] = True

        self.update_boundary_normals()

    def step(self, dt: float, iterations_poisson: int = 12):
        """Advance fluid dynamics by time step dt."""
        self.time += dt
        dx = self.dx
        dy = self.dy
        dx2 = dx * dx
        dy2 = dy * dy
        factor = 0.5 / (dx2 + dy2)

        # 1. Vorticity Advection-Diffusion Step (Explicit upwind / central difference)
        omega_new = [row[:] for row in self.omega]

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid[j][i]:
                    omega_new[j][i] = 0.0
                    continue

                u_val = self.u[j][i]
                v_val = self.v[j][i]

                # Upwind differencing for advection stability
                if u_val > 0:
                    d_omega_dx = (self.omega[j][i] - self.omega[j][i - 1]) / dx
                else:
                    d_omega_dx = (self.omega[j][i + 1] - self.omega[j][i]) / dx

                if v_val > 0:
                    d_omega_dy = (self.omega[j][i] - self.omega[j - 1][i]) / dy
                else:
                    d_omega_dy = (self.omega[j + 1][i] - self.omega[j][i]) / dy

                # Diffusion (central 5-point discrete Laplacian)
                lap_omega = (
                    (self.omega[j][i + 1] - 2.0 * self.omega[j][i] + self.omega[j][i - 1]) / dx2 +
                    (self.omega[j + 1][i] - 2.0 * self.omega[j][i] + self.omega[j - 1][i]) / dy2
                )

                # Time update
                adv = u_val * d_omega_dx + v_val * d_omega_dy
                diff = self.nu * lap_omega
                omega_new[j][i] = self.omega[j][i] + dt * (-adv + diff)

        # Inflow / Outflow / Farfield boundary conditions for vorticity
        for j in range(self.ny):
            omega_new[j][0] = 0.0 # Clean uniform inflow
            # Outflow: zero gradient
            omega_new[j][self.nx - 1] = omega_new[j][self.nx - 2]

        for i in range(self.nx):
            omega_new[0][i] = 0.0
            omega_new[self.ny - 1][i] = 0.0

        self.omega = omega_new

        # 2. Streamfunction Poisson Equation: Lap(psi) = -omega (SOR / Gauss-Seidel)
        omega_sor = 1.35
        for _ in range(iterations_poisson):
            for j in range(1, self.ny - 1):
                y = j * dy
                for i in range(1, self.nx - 1):
                    if self.solid[j][i]:
                        # Solid obstacle boundary: constant streamfunction (impenetrable)
                        self.psi[j][i] = 0.5 * self.u_inf * self.ly
                        continue

                    # Standard Poisson relaxation
                    psi_target = (
                        dy2 * (self.psi[j][i + 1] + self.psi[j][i - 1]) +
                        dx2 * (self.psi[j + 1][i] + self.psi[j - 1][i]) +
                        dx2 * dy2 * self.omega[j][i]
                    ) * factor

                    self.psi[j][i] += omega_sor * (psi_target - self.psi[j][i])

            # Inflow / Outflow / Farfield boundary conditions for psi
            for j in range(self.ny):
                y = j * dy
                self.psi[j][0] = self.u_inf * y # Inflow
                self.psi[j][self.nx - 1] = self.psi[j][self.nx - 2] # Outflow dpsi/dx = 0

            for i in range(self.nx):
                self.psi[0][i] = 0.0 # Bottom slip wall
                self.psi[self.ny - 1][i] = self.u_inf * self.ly # Top slip wall

        # 3. Update Velocities from Streamfunction: u = dpsi/dy, v = -dpsi/dx
        max_v = 0.01
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid[j][i]:
                    self.u[j][i] = 0.0
                    self.v[j][i] = 0.0
                else:
                    self.u[j][i] = (self.psi[j + 1][i] - self.psi[j - 1][i]) / (2.0 * dy)
                    self.v[j][i] = -(self.psi[j][i + 1] - self.psi[j][i - 1]) / (2.0 * dx)

                vel_mag = math.hypot(self.u[j][i], self.v[j][i])
                if vel_mag > max_v:
                    max_v = vel_mag

        self.max_velocity = max_v

        # 4. Enforce Woods Boundary Condition for Solid Wall Vorticity
        # omega_wall = -2 * (psi_fluid - psi_wall) / dn^2
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if self.solid[j][i]:
                    psi_w = self.psi[j][i]
                    # Check 4 neighbors
                    if not self.solid[j][i + 1]:
                        self.omega[j][i] = -2.0 * (self.psi[j][i + 1] - psi_w) / dx2
                    elif not self.solid[j][i - 1]:
                        self.omega[j][i] = -2.0 * (self.psi[j][i - 1] - psi_w) / dx2
                    elif not self.solid[j + 1][i]:
                        self.omega[j][i] = -2.0 * (self.psi[j + 1][i] - psi_w) / dy2
                    elif not self.solid[j - 1][i]:
                        self.omega[j][i] = -2.0 * (self.psi[j - 1][i] - psi_w) / dy2

        # 5. Pressure Poisson Solve & Bernoulli Aerodynamic Recovery
        self.solve_pressure_field(iterations=6)

        # 6. Integrate Aerodynamic Forces (Lift, Drag, Pitching Moment)
        self.integrate_aerodynamic_forces()

        # 7. Advect Flow Tracer Particles (Smoke Streaklines)
        self.update_particles(dt)

    def solve_pressure_field(self, iterations: int = 6):
        """
        Compute gauge pressure field p(x, y) via Pressure Poisson Equation:
        Lap(p) = 2 * rho * (du/dx * dv/dy - du/dy * dv/dx)
        """
        dx = self.dx
        dy = self.dy
        dx2 = dx * dx
        dy2 = dy * dy
        factor = 0.5 / (dx2 + dy2)
        q_dyn = 0.5 * self.rho * (self.u_inf ** 2)

        for _ in range(iterations):
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    if self.solid[j][i]:
                        continue

                    # Velocity derivatives
                    du_dx = (self.u[j][i + 1] - self.u[j][i - 1]) / (2.0 * dx)
                    du_dy = (self.u[j + 1][i] - self.u[j - 1][i]) / (2.0 * dy)
                    dv_dx = (self.v[j][i + 1] - self.v[j][i - 1]) / (2.0 * dx)
                    dv_dy = (self.v[j + 1][i] - self.v[j - 1][i]) / (2.0 * dy)

                    source = 2.0 * self.rho * (du_dx * dv_dy - du_dy * dv_dx)

                    p_target = (
                        dy2 * (self.pressure[j][i + 1] + self.pressure[j][i - 1]) +
                        dx2 * (self.pressure[j + 1][i] + self.pressure[j - 1][i]) -
                        dx2 * dy2 * source
                    ) * factor

                    self.pressure[j][i] = p_target

            # Boundaries
            for j in range(self.ny):
                self.pressure[j][0] = 0.0 # Farfield ambient pressure reference
                self.pressure[j][self.nx - 1] = self.pressure[j][self.nx - 2]
            for i in range(self.nx):
                self.pressure[0][i] = self.pressure[1][i]
                self.pressure[self.ny - 1][i] = self.pressure[self.ny - 2][i]

    def integrate_aerodynamic_forces(self):
        """
        Integrate normal pressure and tangential viscous wall stresses
        around boundary cells to compute Lift and Drag.
        """
        if not self.boundary_cells:
            self.cl = 0.0
            self.cd = 0.0
            self.cm = 0.0
            self.lift_force = 0.0
            self.drag_force = 0.0
            return

        fx_tot = 0.0
        fy_tot = 0.0
        mz_tot = 0.0
        ref_x = 0.5 * self.lx
        ref_y = 0.5 * self.ly

        for bx, by, nx_norm, ny_norm in self.boundary_cells:
            # Pressure on boundary cell
            p_val = self.pressure[by][bx]
            ds = math.hypot(nx_norm * self.dx, ny_norm * self.dy)

            # Normal pressure force
            dFx_press = p_val * nx_norm * ds
            dFy_press = p_val * ny_norm * ds

            # Wall shear stress from vorticity: tau = mu * omega
            omega_w = self.omega[by][bx]
            mu = self.rho * self.nu
            tau = mu * omega_w

            # Tangential vector (perpendicular to normal)
            tx_norm = -ny_norm
            ty_norm = nx_norm
            dFx_visc = tau * tx_norm * ds
            dFy_visc = tau * ty_norm * ds

            dFx = dFx_press + dFx_visc
            dFy = dFy_press + dFy_visc

            fx_tot += dFx
            fy_tot += dFy

            # Moment arm relative to quarter-chord / center
            rx = (bx * self.dx) - ref_x
            ry = (by * self.dy) - ref_y
            mz_tot += (rx * dFy - ry * dFx)

        self.drag_force = fx_tot
        self.lift_force = fy_tot

        # Non-dimensional aerodynamic coefficients
        q_dyn = 0.5 * self.rho * (self.u_inf ** 2) * self.chord_length
        denom = max(1e-6, q_dyn)

        self.cl = self.lift_force / denom
        self.cd = max(0.001, self.drag_force / denom)
        self.cm = mz_tot / (denom * self.chord_length)

    def update_particles(self, dt: float):
        """Advect smoke tracer particles using bilinear interpolated velocity field."""
        self.particle_spawn_timer += dt
        # Periodically spawn smoke particles along vertical rake at x = 0.05 * lx
        if self.particle_spawn_timer >= 0.08:
            self.particle_spawn_timer = 0.0
            num_emitters = 14
            for e in range(num_emitters):
                py = (e + 0.5) / num_emitters * self.ly
                self.particles.append(Particle(0.04 * self.lx, py, max_age=8.0))

        # Advect and filter living particles
        surviving = []
        for p in self.particles:
            p.age += dt
            if not p.is_alive():
                continue

            # Bilinear velocity interpolation
            u_vel, v_vel = self.sample_velocity(p.x, p.y)

            # RK2 midpoint advection
            p.x += u_vel * dt
            p.y += v_vel * dt

            # Domain boundary check
            if 0 <= p.x <= self.lx and 0 <= p.y <= self.ly:
                # Check solid penetration
                gx = int(p.x / self.dx)
                gy = int(p.y / self.dy)
                if 0 <= gx < self.nx and 0 <= gy < self.ny and not self.solid[gy][gx]:
                    surviving.append(p)

        self.particles = surviving

    def sample_velocity(self, x: float, y: float) -> Tuple[float, float]:
        """Sample velocity (u, v) at continuous coordinate via bilinear interpolation."""
        gx = x / self.dx
        gy = y / self.dy

        i0 = max(0, min(self.nx - 2, int(gx)))
        j0 = max(0, min(self.ny - 2, int(gy)))
        i1 = i0 + 1
        j1 = j0 + 1

        fx = max(0.0, min(1.0, gx - i0))
        fy = max(0.0, min(1.0, gy - j0))

        u00 = self.u[j0][i0]; u10 = self.u[j0][i1]
        u01 = self.u[j1][i0]; u11 = self.u[j1][i1]
        u_interp = (1.0 - fx) * (1.0 - fy) * u00 + fx * (1.0 - fy) * u10 + (1.0 - fx) * fy * u01 + fx * fy * u11

        v00 = self.v[j0][i0]; v10 = self.v[j0][i1]
        v01 = self.v[j1][i0]; v11 = self.v[j1][i1]
        v_interp = (1.0 - fx) * (1.0 - fy) * v00 + fx * (1.0 - fy) * v10 + (1.0 - fx) * fy * v01 + fx * fy * v11

        return u_interp, v_interp

    def get_probe_telemetry(self, x: float, y: float) -> Dict[str, float]:
        """Virtual pitot tube telemetry probe at given physical coordinates."""
        gx = max(0, min(self.nx - 1, int(x / self.dx)))
        gy = max(0, min(self.ny - 1, int(y / self.dy)))

        if self.solid[gy][gx]:
            return {
                "x": x,
                "y": y,
                "u": 0.0,
                "v": 0.0,
                "velocity": 0.0,
                "pressure": self.pressure[gy][gx],
                "vorticity": self.omega[gy][gx],
                "streamfunction": self.psi[gy][gx],
                "dynamic_pressure": 0.0,
                "cp": 0.0,
                "is_solid": 1.0
            }

        u_val, v_val = self.sample_velocity(x, y)
        vel_mag = math.hypot(u_val, v_val)

        press_val = self.pressure[gy][gx]
        omega_val = self.omega[gy][gx]
        psi_val = self.psi[gy][gx]

        q_dyn = 0.5 * self.rho * (vel_mag ** 2)
        q_inf = 0.5 * self.rho * (self.u_inf ** 2)
        cp = (press_val - 0.0) / max(1e-6, q_inf)

        return {
            "x": x,
            "y": y,
            "u": u_val,
            "v": v_val,
            "velocity": vel_mag,
            "pressure": press_val,
            "vorticity": omega_val,
            "streamfunction": psi_val,
            "dynamic_pressure": q_dyn,
            "cp": cp,
            "is_solid": 0.0
        }
