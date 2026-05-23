import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import torch

class ROI:    
    def __init__(self, radius, top, bottom, point_density=8000, m=[1,0,0], axis='z'):
        self.radius = radius
        self.top = top
        self.bottom = bottom
        self.m = m
        self.axis = axis

        #points within the ROI
        if(axis == 'z'):
            x = torch.linspace(-radius, radius, int(2*radius*point_density))
            y = torch.linspace(-radius, radius, int(2*radius*point_density))
            z = torch.linspace(bottom, top, int((top-bottom)*point_density))
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            R = torch.sqrt(X**2 + Y**2)
            inside_cylinder = torch.logical_and((R <= self.radius), torch.abs(Z - (self.top + self.bottom)/2) < (self.top - self.bottom)/2)
        elif(axis == 'y'):
            x = torch.linspace(-radius, radius, int(2*radius*point_density))
            z = torch.linspace(-radius, radius, int(2*radius*point_density))
            y = torch.linspace(bottom, top, int((top-bottom)*point_density))
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            R = torch.sqrt(X**2 + Z**2)
            inside_cylinder = torch.logical_and((R <= self.radius), torch.abs(Y - (self.top + self.bottom)/2) < (self.top - self.bottom)/2)
        else:
            y = torch.linspace(-radius, radius, int(2*radius*point_density))
            z = torch.linspace(-radius, radius, int(2*radius*point_density))
            x = torch.linspace(bottom, top, int((top-bottom)*point_density))
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            R = torch.sqrt(Y**2 + Z**2)
            inside_cylinder = torch.logical_and((R <= self.radius), torch.abs(X - (self.top + self.bottom)/2) < (self.top - self.bottom)/2)


        # Apply the mask to filter out points outside the cylinder
        X = torch.where(inside_cylinder, X, torch.nan)
        Y = torch.where(inside_cylinder, Y, torch.nan)
        Z = torch.where(inside_cylinder, Z, torch.nan)

        points = torch.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        self.points = points[~torch.isnan(torch.sum(points, axis=1))]
    
    def integrate(self, coil):
        Bx,By,Bz = coil.biot_savart(self.points)
        return torch.sum(Bx)
    
    def magnetization(self, observation_points, mu_0=4*torch.pi*1e-7):
        def _magnetization(observation_point, mu_0):
            R = observation_point - self.points
            norm_R = torch.linalg.norm(R, axis=1).reshape(-1, 1)
            B = mu_0/(4 * torch.pi) * ((3*R * torch.tile(torch.dot(self.m, R.T),3).reshape(-1,3))/norm_R**5 - self.m/norm_R**3)
            return torch.sum(B,axis=0)
        
        B = torch.zeros((len(observation_points),3))
        for idx, point in enumerate(observation_points):
            B_pt = _magnetization(point, mu_0/len(self.points)/1000000)
            B[idx] = B_pt

        return B[:,0], B[:,1], B[:,2]
    
    def plot(self, ax=None, show=True, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        resolution = 100
            # Generate data for the cylinder
        h = torch.linspace(self.bottom, self.top, resolution)  # Height values
        theta = torch.linspace(0, 2 * torch.pi, resolution)  # Angle values
        theta, h = torch.meshgrid(theta, h, indexing='ij')  # Create a 2D grid for theta and z

        # Parametric equations for the cylinder
        if(self.axis == 'z'):
            x = self.radius * torch.cos(theta)
            y = self.radius * torch.sin(theta)
            z = h
        elif(self.axis == 'y'):
            x = self.radius * torch.cos(theta)
            z = self.radius * torch.sin(theta)
            y = h
        else:
            y = self.radius * torch.cos(theta)
            z = self.radius * torch.sin(theta)
            x = h  

        # Plot the surface of the cylinder
        ax.plot_surface(x, y, z, color='cyan', alpha=0.7, rstride=5, cstride=5)
        # ax.scatter(self.points[:,0],self.points[:,1],self.points[:,2])

        if(show):
            # Customize the plot
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Cylinder')
            axis_size = 1.5*torch.maximum(self.radius, (self.top-self.bottom)/2)
            ax.set_xlim([-axis_size, axis_size])
            ax.set_ylim([-axis_size, axis_size])
            ax.set_zlim([-axis_size, axis_size])
            plt.show()

def plot_magnetic_field(points, Bx, By, Bz, ax=None, show=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    arrowscale = 1
    ax.quiver(points.detach()[:,0], points.detach()[:,1], points.detach()[:,2], Bx*arrowscale, By*arrowscale, Bz*arrowscale, linewidth=0.5)

    if(show): 
        # coil.plot(show=False, ax=ax, color='b', linewidth=2)
        # axis_size = 1.5*torch.maximum(torch.maximum(), (self.top-self.bottom)/2)
        # ax.set_xlim([-grid_size, grid_size])
        # ax.set_ylim([-grid_size, grid_size])
        # ax.set_zlim([-grid_size, grid_size])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Magnetic Field from Coil')

def plotcoilsim(coil, roi, points = 10, ax = None, show=True):
    if(show):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    grid_size = 0.005
    x = torch.linspace(-grid_size/1, grid_size/1, int(points/1))
    y = torch.linspace(-grid_size/5, grid_size/5, int(points/5))
    z = torch.linspace(-grid_size, grid_size, int(points))
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    Bx, By, Bz = torch.zeros_like(X), torch.zeros_like(Y), torch.zeros_like(Z)

    observation_points = torch.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    Bx,By,Bz = coil.biot_savart(observation_points, 1)
    coil.plot(color='b', linewidth=1.5, ax=ax,show=False)
    roi.plot(ax, show=False)
    plot_magnetic_field(observation_points, Bx, By, Bz, ax=ax, show=False)
    ax.set_xlim([-grid_size, grid_size])
    ax.set_ylim([-grid_size, grid_size])
    ax.set_zlim([-grid_size, grid_size])

    if(show):
        plt.show()

def plot_m0(roi, points = 10, ax = None, show=True):
    if(show):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    grid_size = 0.005
    x = torch.linspace(-grid_size/1, grid_size/1, int(points/1))
    y = torch.linspace(-grid_size/1, grid_size/1, int(points/1))
    z = torch.linspace(-grid_size, 0, int(points/2))
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    observation_points = torch.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    Bx,By,Bz = roi.magnetization(observation_points, 1)
    roi.plot(color='b', linewidth=1.5, ax=ax,show=False)
    plot_magnetic_field(observation_points, Bx, By, Bz, ax=ax, show=False)
    ax.set_xlim([-grid_size, grid_size])
    ax.set_ylim([-grid_size, grid_size])
    ax.set_zlim([-grid_size, grid_size])

    if(show):
        plt.show()