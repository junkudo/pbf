import numpy as np
import random
import matplotlib.pyplot as plt
from line_profiler import profile
from scipy.spatial import KDTree
from collections import defaultdict


def poly6_2D(r : float, h : float):

    return np.where((r > h), 0.0, 4.0 / (np.pi * np.power(h,8)) * np.power(h*h - r*r, 3))


def spiky_2D(r : float, h : float):
    if r > h:
        return 0.0
    C = 10.0 / (np.pi * np.power(h, 5))
    return C * np.power((h - r), 3)



def dspikey_2D(r, h):
    r = np.asarray(r)
    C = 10.0 / (np.pi * h**5)

    # Compute the kernel derivative everywhere
    dwdr = -3.0 * C * (h - r)**2

    # Zero out invalid regions (r <= 0 or r > h)
    return np.where((r <= 0.0) | (r > h), 0.0, dwdr)

# def grad_spiky_2D(r_vec : np.ndarray, h : float):
#     r = np.linalg.norm(r_vec)
#     if r <= 0.0 or r > h:
#         return np.zeros_like(r_vec)
#     C = 10.0 / (np.pi * np.power(h,5))
#     dwdr = -3.0 * C * np.power((h-r), 2)
#     return dwdr * r_vec / r


@profile
def sphDensity2D(pi  : np.ndarray,  # shape (d)
               pjs : np.ndarray,  # shape (N,d)
               m : float,
               h : float,
               kernel_func = poly6_2D): # Kernel function of the form kernel_func(r,h)
    # Calculate the density approximation using SPH kernels
    # \sum_j m_j W(pi - pj, h)
    density = 0.0
    r_vec = pi - pjs
    r = np.linalg.norm(r_vec, axis=1)
    density = (m * kernel_func(r,h)).sum(axis=0)
    # for pj in pjs:
    #     r_vec = pi - pj
    #     r = np.linalg.norm(r_vec)
    #     density += m * kernel_func(r, h)

    # Add i the self-particle term
    density += m * kernel_func(0, h)
    return density

@profile
def incompressibilityConstraint(pi : np.ndarray, # shape (d)
                                pjs : np.ndarray, # shape (N,d)
                                rho0 : float,
                                m : float,
                                h : float,
                                kernel_func) :
    rho = sphDensity2D(pi, pjs, m, h, kernel_func)
    return rho / rho0 - 1

@profile
def constraintGradient(pi : np.ndarray,
                       pjs : np.ndarray,
                       rho0 : float,
                       m : float,
                       h : float,
                       dkernel_func) : 
    # Calculate gradient of the ith constraint with respect to the position of ith particle
    # and its N neighbors
    #gradient_i = np.zeros_like(pi)
    # gradient_j = np.zeros_like(pjs)

    # # grad_i = \sum_j grad_rij W(pi - pj) 
    # for pj in pjs:
    #     rij = pi - pj
    #     r = np.linalg.norm(rij)
    #     # We assume r != 0.0 as the neighbors do not include self
    #     # (and the self term has no sensitiivty with respect to pi)


    # grad_j = - grad_rij W()
    rij = pi - pjs
    r = np.linalg.norm(rij, axis=1)


    gradient_i = (dkernel_func(r, h)[:, None] * m / rho0 * rij / r[:, None]).sum(axis = 0)
    gradient_i *= 1.0 / rho0

    gradient_j = -dkernel_func(r,h)[:, None] * m / rho0 * rij / r[:, None]
    # for (j, pj) in enumerate(pjs):
    #     rij = pi - pj
    #     r = np.linalg.norm(rij)
    #     gradient_j[j, :] = -dkernel_func(r,h) * m / rho0 * rij / r

    return (gradient_i, gradient_j)


def generate_particles_2d(width, height, dx):
    xs = np.arange(0, width, dx)
    ys = np.arange(0, height, dx)
    X, Y = np.meshgrid(xs, ys)
    p = np.stack([X.ravel(), Y.ravel()], axis=1)
    return p

def jittered_grid(nx, ny, h, jitter=0.2):
    xs = np.arange(nx) * h
    ys = np.arange(ny) * h
    xv, yv = np.meshgrid(xs, ys)
    pos = np.stack([xv, yv], axis=-1).reshape(-1, 2)
    pos += (np.random.rand(*pos.shape) - 0.5) * h * jitter
    return pos

def testConstraintGradient():
    dim = 2
    num_particles_x = 5
    num_particles_y = 5
    num_particles = num_particles_x * num_particles_y
    # Initialize a 2D matrix of random positions
    # Particle spacing
    dx = 1.0
    # Kernel size to ensure that the particles have enough neighbors to compute properties
    h = 1.8 * dx
    p = jittered_grid(num_particles_x, num_particles_y, dx, dx * 0.05)

    rho0 = 1.0
    # Mass of the particles to match the density with the particle spacing
    m = rho0 * dx * dx * np.random.rand(num_particles)


    for i in range(p.shape[0]):
        js = getNeighborsWithinDistance(i, p, h)

        # For each particle, calculate lagrange multiplier
        pi = p[i, :]
        pjs = p[js,:]



        
        [dcdpi, dcdpj] = constraintGradient(pi, pjs, rho0, m, h, dspikey_2D)

        # Test the gradient with respect to pi

        dcdp_numerical = np.zeros_like(pi)

        for d in range(len(pi)):
            eps = 1.0e-6
            pi_plus = pi.copy()
            pi_minus = pi.copy()
            pi_plus[d] += eps
            pi_minus[d] -= eps
            c_plus = incompressibilityConstraint(pi_plus, pjs, rho0, m, h, spiky_2D)
            c_minus = incompressibilityConstraint(pi_minus, pjs, rho0, m, h, spiky_2D)
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
                c_plus = incompressibilityConstraint(pi, pjs_plus, rho0, m, h, spiky_2D)
                c_minus = incompressibilityConstraint(pi, pjs_minus, rho0, m, h, spiky_2D)            
                dcdp = (c_plus - c_minus) / (2.0 * eps)
                dcdpj_numerical[k, d] = dcdp

        assert np.allclose(dcdpj_numerical, dcdpj)



    # print(incompressibilityConstraint(pi, pjs, rho0, m, h, poly6_2D))
    # print(constraintGradient(pi, pjs, rho0, m, h, grad_spiky_2D))
def calculateGravity2D(p, m : float, g : float):
    # We are assuming gravity in 2D is of the form (0.0, -g)
    f = np.zeros_like(p)
    f[:,1] = -m*g
    return f

def updateVelocity(v : np.ndarray, f : np.ndarray, deltaT : float, m : float) :
    vstar = v + f / m * deltaT
    return vstar

def predictPositionFromForces(p : np.ndarray, v : np.ndarray, deltaT : float):
    pstar = p + deltaT * v
    return pstar

@profile
def getNeighborsWithinDistance(i : int, p : np.ndarray, d : float):
    # Returns list of neighbors within a certain distance by index

    pi = p[i,:]
    neighbors = []
    for j in range(p.shape[0]):
        if i == j:
            continue
        pj = p[j,:]
        mag = np.linalg.norm(pi - pj)

        if (mag < d):
            neighbors.append(j)

    return neighbors


def build_grid(positions, cell_size):
    grid = defaultdict(list)
    for i, p in enumerate(positions):
        key = tuple((p // cell_size).astype(int))
        grid[key].append(i)
    return grid

def get_neighbors(i, positions, grid, cell_size):
    p = positions[i]
    key = tuple((p // cell_size).astype(int))
    ix, iy = key
    
    neigh = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cell = (ix+dx, iy+dy)
            for j in grid.get(cell, []):
                if j != i:
                    neigh.append(j)
    return neigh

@profile
def run():

    # ---
    # Initialization
    dim = 2
    num_particles_x = 10
    num_particles_y = 10
    num_active_particles = num_particles_x * num_particles_y


    bottom_wall_location_y = 0
    num_wall_particles_y = 3
    num_wall_particles_x = 10

    # Initialize a 2D matrix of random positions that have expected densities equal to the rest densities
    # for the interior particles
    
    # Target particle spacing
    dx = 0.01

    
    # Kernel size to ensure that the particles have enough neighbors to compute properties (this gives me 20 ish)
    h = 2.5 * dx
    p = jittered_grid(num_particles_x, num_particles_y, dx, dx * 0.05) + [0, 0.05]

    # p_right_wall = jittered_grid(3, 20, dx, 0.0) + [10 * dx , 0.0]

    # p_wall = jittered_grid(num_wall_particles_x, num_wall_particles_y, dx, 0.0)

    # p = np.concatenate((p, p_wall))

    # p_left_wall = jittered_grid(3, 20, dx, 0.0) - [dx * 3, 0.0]
    # p = np.concatenate((p, p_left_wall))

    # p_right_wall = jittered_grid(3, 20, dx, 0.0) + [10 * dx , 0.0]
    # p = np.concatenate((p, p_right_wall))

    # Compliance (inverse stiffness)
    k = 1.0e-6
    alpha = 1.0 / k


    rho0 = 1.0
    # Mass of the particles to match the density with the particle spacing
    m = rho0 * dx * dx



    
    g = 9.8
    f = calculateGravity2D(p, m, g)

    # Start at zero velocity
    v = np.zeros_like(p)
    
    # ----
    # Initialization of pyplot
    plt.ion()
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], s=5)
    delta = .10 * (p[:,0].max() - p[:,0].min())
    ax.set_xlim(p[:,0].min() - delta, p[:,0].max() + delta)
    ax.set_ylim(-.1, p[:,1].max() + delta)
    ax.set_aspect('equal')

    # ----
    # Simulation loop
    deltaT = 1 / 240 # I have no idea what unity's physics time step is

    time = 0

    for simulationStep in range(30):
        # Lagrange Storage of lambdas (1 per active particle)
        lambdas = np.zeros(num_active_particles)
        
        time += deltaT

        v = updateVelocity(v, f, deltaT, m)
        v[num_active_particles:] = 0.0 # Zero out the velocity for the non-active particles

        pstar = predictPositionFromForces(p, v, deltaT)


        grid = build_grid(pstar, h)
        neighbors = []
        for i in range(num_active_particles):
            neighbors.append(get_neighbors(i, pstar, grid, h))

        niter = 50

        # For each solver iteration,
        for solverIter in range(niter):
            avgAbsCi = 0

            # For each constraint,
            for i in range(num_active_particles):
                js = neighbors[i]
                
                # ---
                # Calculate delta in lagrange multiplier
                pi = pstar[i, :]
                pjs = pstar[js,:]

                ci = incompressibilityConstraint(pi, pjs, rho0, m, h, poly6_2D)

                avgAbsCi += np.abs(ci)
                [gradici, gradjci] = constraintGradient(pi, pjs, rho0, m, h, dspikey_2D)

                # Sum of the squared norms of the gradients
                normSqed = 0.0
                normSqed += np.dot(gradici, gradici)
                for grad in gradjci:
                    normSqed += np.dot(grad,grad)

                # delta lambda for this constraint
                dlambda = (-ci - alpha * lambdas[i]) / (1 / m * normSqed + alpha)


                # apply delta x to all neighbors of this constraint based on incremental delta lambda
                pstar[i, :] += 1 / m * gradici * dlambda






                # -- 
                # Artifical pressure term
                # We calculate the contribution to all the neighbors to follow the pattern set 
                # by the lambda_j contributions.  We calculate pj - pi to reflect this change
                rij = pjs - pi 
                r = np.linalg.norm(rij, axis=1)

                k = 0.001
                q = 0.2 * h
                n = 4
                scorr = -k * np.pow((poly6_2D(r, h) / poly6_2D(q, h)), n)

                pstar[js, :] += 1 / m * gradjci * (dlambda + scorr[:, None])
                # i can probably sum  the contributions to my neighgbors or go the other way around again
                # (if i flip r), or sum  into myself by flipping the graidents
                # for (j, gradj) in zip(js, gradjci):
                #     # Add articial pressure term to neighbors
                #     k = 0.001
                #     q = 0.2 * h
                #     n = 4
                #     scorr = -k * np.pow((poly6_2D(r, h) / poly6_2D(q, h)), n)

                #     pstar[j, :] += 1 / m * gradj * (dlambda + scorr)


                lambdas[i] += dlambda

                # Artificial pressure?

            # deltap = np.zeros_like(p)

            # for i in range(num_active_particles):
            #     # js = [neighbors[i, j] for j in range(30) if distances[i, j] < h and distances[i,j] > 0.0]
            #     js = np.asarray(neighbors[i])

            #     pi = pstar[i, :]
            #     pjs = pstar[js,:]

            #     lambdai = lambdas[i]
            #     mask = js < num_active_particles           # only valid neighbors

            #     lambdajs = np.zeros(js.shape, dtype=float)
            #     lambdajs[mask] = lambdas[js[mask]]         # safe: all indices in bounds


            #     rji = pi - pjs
            #     r = np.linalg.norm(rji, axis=1)
            #     gradW = dspikey_2D(r, h)[:, None] * rji / r[:,None]


            #     # Calculate artificial pressure term
            #     k = 0.001
            #     q = 0.2 * h
            #     n = 4
            #     scorr = -k * np.pow((poly6_2D(r, h) / poly6_2D(q, h)), n)
            #     deltap[i , :] = (m / rho0 * (lambdai + lambdajs[:,None] + scorr[:,None]) * gradW).sum(axis=0)


                # for j in js:
                #     pj = pstar[j,:]
                #     lambdai = lambdas[i]
                #     lambdaj = lambdas[j] if (j < num_active_particles) else 0.0
                #     rji = pi - pj
                #     r = np.linalg.norm(rji) 
                #     gradW = dspikey_2D(r, h) * rji / r

                #     # Calculate artificial pressure term
                #     k = 0.001
                #     q = 0.2 * h
                #     n = 4
                #     scorr = -k * np.pow((poly6_2D(r, h) / poly6_2D(q, h)), n)
                #     # print(scorr)
                #     deltap[i,:] += m / rho0 * (lambdai + lambdaj + scorr) * gradW

            # Note - I'm not sure if I need this under-relaxation term
            # pstar += 0.1 * deltap
            avgAbsCi /= num_active_particles
            print(avgAbsCi)
            # print(avgAbsCi)
        # Update velocity
        v = (pstar - p) / deltaT

        # I hate python copy by reference semantics.  I'm not sure when I'm getting new numpy arrays.
        p = pstar.copy()
        

        # Apply vorticity confinement and XSPH viscosity
        # XSPH Viscosity 
        c = 0.0001
        vcorrections = np.zeros_like(v)
        for i in range(num_active_particles):
            # js = [neighbors[i, j] for j in range(30) if distances[i, j] < h and distances[i,j] > 0.0]
            js = neighbors[i]
            # js.sort()
            # jun = np.all(js == neighbors2[i])
            # if (not jun):
            #     print(i)
            #     print(js)
            #     print(neighbors2[i])
            #     raise KeyError

            pi = p[i,:]
            vi = v[i, :]
            pjs = p[js,:]
            vjs = v[js,:]
            rji = pi - pjs
            vij = vjs - vi
            vcorrections[i] = np.sum(c * m *  vij  * (poly6_2D(np.linalg.norm(pi - pjs, axis=1), h)[:,None]), axis=0)

            # for j in js:
            #     pj = p[j,:]
            #     vj = v[j,:]

            #     rji = pi - pj
            #     vij = vj - vi
            #     # Something is wrong with this
            #     # Does this need the mass term?  maybe
            #     vcorrections[i,:] += c * m *  vij  * poly6_2D(np.linalg.norm(pi - pj), h)

        v += vcorrections

        # --- update plot ---
        scat.set_offsets(p)
        frame = 0
        ax.set_title(f"Frame {simulationStep}")
        plt.pause(0.001)

    #plt.ioff()   # turn off interactive mode
    #plt.show()   # keeps the window open



if __name__ == "__main__":
    run()