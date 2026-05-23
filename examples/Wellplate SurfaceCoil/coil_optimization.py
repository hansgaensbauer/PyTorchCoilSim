import coilsim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

if __name__ == "__main__":
    roi_rad = 0.0034
    roi_top = 0.0015
    roi_bottom = 0.00075
    roi = coilsim.ROI(roi_rad, roi_top, roi_bottom)
    # plot_m0(roi)

    ###############################################################'
    ##                Coil Comparative Simulations
    ###############################################################

    coilsfig, ax = plt.subplots(1,3, subplot_kw=dict(projection='3d'))
    print("Box: ", end="")
    coil = coilsim.Coil.boxspiral(0.0016, 0.008, 0.01, 20, 10000, center=(0,0,-0.0008),axis='z')
    print((roi.integrate(coil)/coil.get_length()).item())
    coilsim.plotcoilsim(coil,roi,ax=ax[0], show=False)

    print("D: ", end="")
    points_per_turn = 1000
    coil = coilsim.Coil.d(radius=0.005, points_per_turn=points_per_turn, center=(0,0,0))
    print((roi.integrate(coil)/coil.get_length()).item())
    coilsim.plotcoilsim(coil,roi,ax=ax[1], show=False)

    print("Spiral: ", end="")
    coil = coilsim.Coil.spiral(radius=0.0015, pitch=0.0002, turns=8, center=(0.002,0,0))
    print((roi.integrate(coil)/coil.get_length()).item())
    coilsim.plotcoilsim(coil,roi,ax=ax[2], show=False)

    ###############################################################'
    ##                Surface Flux Optimization
    ###############################################################

    #define the coil plane
    grid_size = 0.01
    points = 50
    x = np.linspace(-grid_size, grid_size, points)
    y = np.linspace(-grid_size, grid_size, points)
    z = np.linspace(0, 0, 1)
    X, Y, Z = np.meshgrid(x, y, z)
    Bx, By, Bz = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    observation_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    Bx,By,Bz = roi.magnetization(observation_points, 1)
    zmag = Bz.reshape(len(x),-1)

    #Plot Z flux over the plane
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(x, y, zmag)
    fig.colorbar(pcm, ax=ax)
    ellipse = Ellipse(xy=(0, 0), width=0.00865, height=0.00687, 
                        edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.vlines(x=0,ymin=-0.00687/2,ymax=0.00687/2,color='r', linewidth=2)
    plt.show()

    def inside(x,y,length,width):
        return np.sqrt(x**2/(length/2)**2 + y**2/(width/2)**2) < 1
    
    def ellipse_perimeter(a, b): #(sigh)
        # Ramanujan's second approximation
        h = (3 * (a + b)) - np.sqrt((3 * a + b) * (a + 3 * b))
        return np.pi * h

    #Optimize seperately for X and Y coil sizes (length and width)
    lengths = np.linspace(grid_size/(points/2),2*grid_size, int(points/2))
    widths = np.linspace(grid_size/(points/2),2*grid_size, int(points/2))
    phi = np.zeros((len(lengths),len(widths)))
    for i in range(len(lengths)):
        for j in range(len(widths)):
            for x_idx in range(len(x)):
                for y_idx in range(len(y)):
                    if(inside(x[x_idx],y[y_idx], lengths[i], widths[j])):
                        if(y[y_idx] < 0):
                            phi[i,j] += zmag[x_idx, y_idx]
                        elif(y[y_idx] > 0):
                            phi[i,j] -= zmag[x_idx, y_idx]   
                        else:
                            print("zero!")
            #weight by coil length
            phi[i,j] = phi[i,j]/(ellipse_perimeter(lengths[i], widths[j]) + lengths[i]*2)

        #Plot Z flux over the plane
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(lengths, widths, phi)
    plt.xlabel("Length (m)")
    plt.ylabel("Width (m)")
    fig.colorbar(pcm, ax=ax)
    plt.show()