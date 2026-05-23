import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

class Coil:
    def __init__(self, points):
        self.points = points  # Shape: (N, 3)

    @classmethod
    def circular(cls, radius=1.0, turns=1, points_per_turn=100, center=(0, 0, 0), axis='z'):
        total_points = turns * points_per_turn
        theta = torch.linspace(0, 2 * torch.pi * turns, total_points)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z = torch.zeros_like(theta)

        if axis == 'x':
            points = torch.vstack((z, x, y)).T
        elif axis == 'y':
            points = torch.vstack((x, z, y)).T
        else:
            points = torch.vstack((x, y, z)).T

        points += torch.array(center)
        return cls(points)
    
    @classmethod
    def spiral(cls, radius=1.0, turns=1, pitch=0.001, points_per_turn=100, center=(0, 0.2, 0), axis='z'):
        total_points = turns * points_per_turn
        theta = torch.linspace(0, 2 * torch.pi * turns, total_points)
        x = torch.cos(-theta)*(theta+radius/pitch * 2 * torch.pi)/(2*torch.pi)*pitch
        y = torch.sin(-theta)*(theta+radius/pitch * 2 * torch.pi)/(2*torch.pi)*pitch
        z = torch.zeros_like(theta)

        if axis == 'x':
            points = torch.vstack((z, x, y)).T
        elif axis == 'y':
            points = torch.vstack((x, z, y)).T
        else:
            points = torch.vstack((x, y, z)).T

        points += torch.array(center)
        return cls(points)

    @classmethod
    def boxspiral(cls, height, width, length, turns=10, points_per_meter=10000, center=(0,0,0), axis='z'):
        riser_points = int(height * points_per_meter)
        horizontal_points = int(width * points_per_meter)
        diagonal_points = int(torch.sqrt((length/turns)**2 + width**2)*points_per_meter)
        points_per_turn = 2 * riser_points + horizontal_points + diagonal_points
        total_points = turns * points_per_turn
        x = torch.empty(total_points)
        y = torch.empty(total_points)
        z = torch.empty(total_points)

        for i in range(turns):
            turn_start_idx = i*points_per_turn
            #First vertical segment
            x[turn_start_idx:turn_start_idx + riser_points] = torch.zeros(riser_points) - length/2 + (length/turns*i)
            y[turn_start_idx:turn_start_idx + riser_points] = torch.zeros(riser_points) - width/2
            z[turn_start_idx:turn_start_idx + riser_points] = torch.linspace(-height/2, height/2, riser_points)
            #Horizontal segment
            x[turn_start_idx + riser_points:turn_start_idx + riser_points + horizontal_points] = torch.zeros(horizontal_points) - length/2 + (length/turns*i)
            y[turn_start_idx + riser_points:turn_start_idx + riser_points + horizontal_points] = torch.linspace(-width/2, width/2, horizontal_points)
            z[turn_start_idx + riser_points:turn_start_idx + riser_points + horizontal_points] = torch.zeros(horizontal_points) + height/2
            #Second vertical segment
            x[turn_start_idx + riser_points + horizontal_points:turn_start_idx + 2*riser_points + horizontal_points] = torch.zeros(riser_points) - length/2 + (length/turns*i)
            y[turn_start_idx + riser_points + horizontal_points:turn_start_idx + 2*riser_points + horizontal_points] = torch.zeros(riser_points) + width/2
            z[turn_start_idx + riser_points + horizontal_points:turn_start_idx + 2*riser_points + horizontal_points] = torch.linspace(height/2, -height/2, riser_points)
            #Diagonal segment
            x[turn_start_idx + 2*riser_points + horizontal_points:turn_start_idx + points_per_turn] = torch.linspace(-length/2 + (length/turns*i), -length/2 + (length/turns*(i+1)), diagonal_points)
            y[turn_start_idx + 2*riser_points + horizontal_points:turn_start_idx + points_per_turn] = torch.linspace(width/2, -width/2, diagonal_points)
            z[turn_start_idx + 2*riser_points + horizontal_points:turn_start_idx + points_per_turn] = torch.zeros(diagonal_points) - height/2
        
        if axis == 'x':
            points = torch.vstack((z, x, y)).T
        elif axis == 'y':
            points = torch.vstack((x, z, y)).T
        else:
            points = torch.vstack((x, y, z)).T

        points += torch.array(center)
        return cls(points)
    
    @classmethod
    def d(cls, radius=1.0, turns=1, points_per_turn=100, center=(0, 0, 0), axis='z'):
        total_points = turns * points_per_turn
        theta = torch.linspace(0, 2 * torch.pi * turns, total_points)
        x_firsthalf = radius * torch.cos(theta[:int(points_per_turn/2)])
        y_firsthalf = radius * torch.sin(theta[:int(points_per_turn/2)])
        z_firsthalf = torch.zeros_like(y_firsthalf)
        x_cross = torch.linspace(-radius, radius, int(points_per_turn/torch.pi))
        y_cross = torch.zeros_like(x_cross)
        z_cross = torch.zeros_like(x_cross)
        x_secondhalf = -radius * torch.cos(theta[int(points_per_turn/2):])
        y_secondhalf = radius * torch.sin(theta[int(points_per_turn/2):])
        z_secondhalf = torch.zeros_like(y_secondhalf)
        x_returncross = torch.linspace(-radius, radius, int(points_per_turn/torch.pi))
        y_returncross = torch.zeros_like(x_cross)
        z_returncross = torch.zeros_like(x_cross)

        x = torch.concatenate((x_firsthalf, x_cross, x_secondhalf, x_returncross))
        y = torch.concatenate((y_firsthalf, y_cross, y_secondhalf, y_returncross))
        z = torch.concatenate((z_firsthalf, z_cross, z_secondhalf, z_returncross))

        if axis == 'x':
            points = torch.vstack((z, x, y)).T
        elif axis == 'y':
            points = torch.vstack((x, z, y)).T
        else:
            points = torch.vstack((y, x, z)).T #WARNING! This is swapped

        points += torch.array(center)
        return cls(points)

    def get_segments(self):
        segments = torch.stack((self.points[:-1], self.points[1:]), axis=1)
        return segments

    def get_current_elements(self, I=1.0):
        segments = self.get_segments()
        dl = segments[:, 1] - segments[:, 0]
        r = (segments[:, 0] + segments[:, 1]) / 2
        return dl, r, I
    
    def get_length(self):
        segments = self.get_segments()
        dl = segments[:, 1] - segments[:, 0]
        return torch.sum(torch.linalg.vector_norm(dl, dim=1))

    def plot(self, show=True, ax=None, **kwargs):
        def _set_axes_equal(ax):
            """
            Make axes of 3D plot have equal scale so that spheres appear as spheres,
            cubes as cubes, etc.

            Input
            ax: a matplotlib axis, e.g., as output from plt.gca().
            """

            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = torch.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = torch.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = torch.mean(z_limits)

            # The plot bounding box is a sphere in the sense of the infinity
            # norm, hence I call half the max range the plot radius.
            plot_radius = 0.5*max([x_range, y_range, z_range])

            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.points.detach()[:, 0], self.points.detach()[:, 1], self.points.detach()[:, 2], **kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if show:
            if ax is None:
                _set_axes_equal(ax)
                ax.set_box_aspect([1.0, 1.0, 1.0])
            plt.show()

    def biot_savart(self, points, I=1.0, mu=4*torch.pi*1e-7):
        dl, r, current = self.get_current_elements(1)
        R = points.unsqueeze(1) - r.unsqueeze(0)
        norm_R = torch.cdist(points, r, p=2)
        cross = torch.cross(R, dl.unsqueeze(0), dim=2)
        dB = (mu / (4 * torch.pi)) * (current * cross) 
        dB = torch.einsum('pst,ps->pst', dB, norm_R**(-3))
        B = torch.sum(dB, axis=1)
        return B
