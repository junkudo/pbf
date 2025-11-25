import pbf
import numpy as np

def integrate_kernel(kernel, h, N=200000):
    rs = np.linspace(0, h, N)
    dr = rs[1] - rs[0]
    values = kernel(rs, h)
    # 2D integration: 2π r W(r)
    integrand = 2 * np.pi * rs * values
    return np.sum(integrand) * dr

def test_kernel_coefficients_2D():
    h = 3.0
    kernels = [pbf.poly6_2D, pbf.spiky_2D]
    tol = 1.0e-5

    for kernel in kernels:
        val = integrate_kernel(kernel, h)
        assert abs(val - 1) < 1.0e-5, f"Assertion failed: value={val} not close to one"

def test_spiky_gradient():
    h = 4.2
    r = np.random.rand()
    func = pbf.spiky_2D
    gradient = pbf.dspikey_2D
    grad = gradient(r,h)

    eps = 1.0e-5
    rplus  = r + eps
    rminus = r - eps
    fplus = func(rplus, h)
    fminus = func(rminus, h)

    num_grad = (fplus - fminus) / (2 * eps)
    tol = 1.0e-5
    assert abs(grad - num_grad) < tol, f"Numerical gradient not matching analytical"


def test_constraint_gradient():
    dim = 2
    num_particles_x = 5
    num_particles_y = 5
    num_particles = num_particles_x * num_particles_y
    # Initialize a 2D matrix of random positions
    # Particle spacing
    dx = 1.0
    # Kernel size to ensure that the particles have enough neighbors to compute properties
    h = 1.8 * dx
    p = pbf.jittered_grid(num_particles_x, num_particles_y, dx, dx * 0.05)

    rho0 = 1.0
    # Mass of the particles to match the density with the particle spacing
    m = rho0 * dx * dx


    for i in range(p.shape[0]):

        js = pbf.getNeighborsWithinDistance(i, p, h)

        # For each particle, calculate lagrange multiplier
        pi = p[i, :]
        pjs = p[js,:]
        print(np.shape(pjs))

        [dcdpi, dcdpj] = pbf.constraintGradient(pi, pjs, rho0, m, h, pbf.dspikey_2D)

        # Test the gradient with respect to pi
        dcdp_numerical = np.zeros_like(pi)

        for d in range(len(pi)):
            eps = 1.0e-6
            pi_plus = pi.copy()
            pi_minus = pi.copy()
            pi_plus[d] += eps
            pi_minus[d] -= eps
            c_plus = pbf.incompressibilityConstraint(pi_plus, pjs, rho0, m, h, pbf.spiky_2D)
            c_minus = pbf.incompressibilityConstraint(pi_minus, pjs, rho0, m, h, pbf.spiky_2D)
            dcdp = (c_plus - c_minus) / (2.0 * eps)
            dcdp_numerical[d] = dcdp
        assert np.allclose(dcdp_numerical, dcdpi)

        dcdpj_numerical = np.zeros_like(pjs)
        for k in range(len(pjs)):
            for d in range(pjs.shape[1]):
                pjs_plus = pjs.copy()
                pjs_minus = pjs.copy()
                eps = 1.0e-6
                pjs_plus[k, d] += eps
                pjs_minus[k, d] -= eps
                c_plus = pbf.incompressibilityConstraint(pi, pjs_plus, rho0, m, h, pbf.spiky_2D)
                c_minus = pbf.incompressibilityConstraint(pi, pjs_minus, rho0, m, h, pbf.spiky_2D)            
                dcdp = (c_plus - c_minus) / (2.0 * eps)
                dcdpj_numerical[k, d] = dcdp
        assert np.allclose(dcdpj_numerical, dcdpj)

def run_tests():
    test_kernel_coefficients_2D()
    test_spiky_gradient()
    test_constraint_gradient()

if __name__ == "__main__":
    run_tests()