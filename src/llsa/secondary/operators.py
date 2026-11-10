"""Frozen-radius cylindrical equations, with an explicit zero-harmonic closure."""
import numpy as np

from .discretization import inertial_base, pier_grid, rotation_rates


class PierOperators:
    def __init__(self, radius=510.0, beta=68.0, n=40, a=2.0, b=.6, c=.5,
                 mean_mode="cylindrical", ro=-1.):
        if not np.isfinite(radius) or radius <= 0 or not np.isfinite(beta):
            raise ValueError("Positive finite radius and real finite beta required")
        self.radius, self.beta = float(radius), float(beta)
        self.disk_rotation, self.external_rotation = rotation_rates(ro)
        self.ro = float(ro)
        self.re = abs(self.ro)*self.radius
        if mean_mode not in ("parallel", "cylindrical"):
            raise ValueError("mean_mode must be parallel or cylindrical")
        self.mean_mode = mean_mode
        self.grid = pier_grid(n, a, b, c)
        self.base = inertial_base(n, a, b, c, self.ro)
        self.m = m = self.grid.size
        self.size = 4*m
        self.I = np.eye(m)
        self.boundary_rows = np.array([v*m+j for v in range(4) for j in (0, m-1)])
        self.M = np.diag(np.r_[np.ones(3*m), np.zeros(m)])
        self.M[self.boundary_rows] = 0

    def linear(self, k, ell, mean_closure=True):
        """A q = i*(omega/R)*M q, components stored [u(z),v(z),w(z),p(z)].

        Velocities and pressure are scaled by R and R^2 internally; alpha and beta
        retain Pier's convention. The boundary pressure condition is p_z=w_zz/R.
        """
        R, I, m = self.radius, self.I, self.m
        D, D2 = self.grid.D, self.grid.D2
        U, V, W, dU, dV, dW = self.base
        scalar = np.diag(1j*k*U+1j*ell*V/R+(k*k+ell*ell/R**2-1j*k/R)/R)
        scalar += W[:, None]*D/R-D2/R
        A = np.zeros((4*m, 4*m), complex)
        for v in range(3):
            correction = dW/R if v == 2 else U/R+1/R**3
            A[v*m:(v+1)*m, v*m:(v+1)*m] = scalar+np.diag(correction)
        A[:m, m:2*m] = np.diag(-2*V/R+2j*ell/R**3)
        A[m:2*m, :m] = np.diag(2*V/R-2j*ell/R**3)
        A[:m, 2*m:3*m] = np.diag(dU)
        A[m:2*m, 2*m:3*m] = np.diag(dV)
        A[:m, 3*m:] = 1j*k*I
        A[m:2*m, 3*m:] = 1j*ell/R*I
        A[2*m:3*m, 3*m:] = D
        A[3*m:, :m] = (1j*k+1/R)*I
        A[3*m:, m:2*m] = 1j*ell/R*I
        A[3*m:, 2*m:3*m] = D
        A[self.boundary_rows] = 0
        for v in range(3):
            for j in (0, m-1):
                A[v*m+j, v*m+j] = 1
        A[3*m, 3*m:] = D[0]
        A[3*m, 2*m:3*m] = -D2[0]/R
        A[-1, -1] = 1
        if mean_closure and k == 0 and ell == 0:
            if self.mean_mode == "parallel":
                # Parallel mean-flow correction: w_0=0. This is an explicit
                # modelling choice, assessed against Pier's Table 2, not implied
                # by retaining u_0/R in the frozen cylindrical continuity law.
                A[3*m+1:4*m-1] = 0
                A[3*m+1:4*m-1, 2*m:3*m] = I[1:-1]
            else:
                # Full mean continuity w_0'=-u_0/R needs a free axial value at
                # infinity. Imposing both w_0 endpoints creates a checkerboard
                # mode in equal-order primitive-variable collocation.
                A[3*m-1] = 0
                A[3*m-1, 2*m:3*m] = D[0]
        return A

    def coupling(self, base_mode, base_number, alpha, k, ell):
        """Linearization of nonlinear advection about one primary Fourier mode.

        The input perturbation has wavevector (k,ell); the base harmonic has
        wavevector (base_number*alpha, base_number*beta). Output harmonic is their sum.
        """
        R, m, D = self.radius, self.m, self.grid.D
        u = np.asarray(base_mode).reshape(4, m)[:3]
        du = u@D.T
        C = np.zeros((4*m, 4*m), complex)
        adv = np.diag(1j*k*u[0]+1j*ell/R*u[1])+u[2, :, None]*D
        for v in range(3):
            row = slice(v*m, (v+1)*m)
            C[row, row] += adv
            C[row, :m] += np.diag(1j*base_number*alpha*u[v])
            C[row, m:2*m] += np.diag(1j*base_number*self.beta/R*u[v])
            C[row, 2*m:3*m] += np.diag(du[v])
        C[:m, m:2*m] -= np.diag(2*u[1]/R)
        C[m:2*m, :m] += np.diag(u[1]/R)
        C[m:2*m, m:2*m] += np.diag(u[0]/R)
        C[self.boundary_rows] = 0
        return C

    def nonlinear(self, modes, alpha):
        """Exact truncated convolution; modes indexed -H,...,+H. No FFT aliasing."""
        H = (len(modes)-1)//2
        q = np.asarray(modes).reshape(2*H+1, 4, self.m)
        dq = q[:, :3]@self.grid.D.T
        out = np.zeros_like(q)
        for out_n in range(-H, H+1):
            for j in range(-H, H+1):
                k = out_n-j
                if abs(k) > H:
                    continue
                v, w = q[j+H], q[k+H]
                adv = 1j*k*(alpha*v[0]+self.beta/self.radius*v[1])
                out[out_n+H, :3] += adv*w[:3]+v[2]*dq[k+H]
                out[out_n+H, 0] -= v[1]*w[1]/self.radius
                out[out_n+H, 1] += v[0]*w[1]/self.radius
        out = out.reshape(len(modes), self.size)
        out[:, self.boundary_rows] = 0
        return out
