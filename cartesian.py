import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy.polynomial.legendre as lg
from scipy.integrate import dblquad


# Numerical integration via Gaussian Quadrature

def GaussQuad(f, xi, xf, n, M):
    I = 0
    c = np.append(np.zeros(M),1)
    x = lg.legroots(c)
    w = 0*x
    d = lg.legder(c)
    for i in range(len(x)):
        w[i] = 2./((1-x[i]**2)*(lg.legval(x[i],d))**2)
    for i in range(M):
        I += w[i]*f(n,interval(x[i], xf, xi))
    I *= 0.5*(xf - xi);

    return I
  
def interval(x, xi, xf):
  return 0.5*(xf + xi + x*(xf-xi))
  
  
# Define main functions

Lx, Ly = 10, 10
c = 100

k_x = lambda n: np.pi*n/Lx
k_y = lambda m: np.pi*m/Ly
eigenf_x = lambda n,x: np.sin(k_x(n)*x)
eigenf_y = lambda m,y: np.sin(k_y(m)*y)
f = lambda n,m,x,y,temp: 2/np.sqrt(Lx*Ly)*temp*eigenf_x(n,x)*eigenf_y(m,y)
k = lambda n,m: np.sqrt(k_x(n)**2 + k_y(m)**2)

n_max, m_max = 10, 10

# Build the meshgrid

N = 150    # Meshsize
fps = 10    # Frames per sec
frn = 50   # Frame number of the animation

x = np.linspace(0, Lx, N+1)
y = np.linspace(0, Ly, N+1)
x, y = np.meshgrid(x, y)
zarray = np.zeros((N+1, N+1, frn))


# Calculate time coefficients and final solution

init_pos = lambda y,x: np.sin(x)
init_vel = lambda y,x: np.cos(x)
norm_x = GaussQuad(lambda n,x: eigenf_x(n,x)**2, 0, Lx, 1, 3)
norm_y = GaussQuad(lambda m,y: eigenf_y(m,y)**2, 0, Ly, 1, 3)

A, B = np.zeros((n_max, m_max)), np.zeros((n_max, m_max))
nvec, mvec = np.meshgrid(range(1,n_max), range(1,m_max))
for n,m in zip(nvec.flatten(),mvec.flatten()):
    int_pos = lambda y,x: init_pos(y,x)*eigenf_x(n,x)*eigenf_y(m,y)
    init_cond_pos = dblquad(int_pos, 0, Lx, 0, Ly)
    int_vel = lambda y,x: init_vel(y,x)*eigenf_x(n,x)*eigenf_y(m,y)
    init_cond_vel = dblquad(int_vel, 0, Lx, 0, Ly)
    A[n,m] = init_cond_pos[0]/(norm_x*norm_y)
    B[n,m] = (1/(c*k(n,m)))*init_cond_vel[0]/(norm_x*norm_y)

for t in range(frn):
    for n,m in zip(nvec.flatten(),mvec.flatten()):
        zarray[:,:,t] += f(n,m,x,y,A[n,m]*np.cos(c*k(n,m)*t) + B[n,m]*np.sin(c*k(n,m)*t))

    
# Draw animation  
    
def update_plot(frame_number, zarray, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(x, y, zarray[:,:,frame_number], cmap = "magma", clim = (-0.3,0.3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot = [ax.plot_surface(x, y, zarray[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(0,1.5)
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ani = anim.FuncAnimation(fig, update_plot, frn, fargs=(zarray, plot), interval=1000/fps)

plt.show()


# Save the animation

# title = "title"
# ani.save(title + '.gif', writer = 'imagemagick', fps = fps)
