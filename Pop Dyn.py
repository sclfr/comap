import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

class ParticleSimulation:
    def __init__(self, Lx=20, Ly=10, corridor_width=2, num_left=5, num_right=5):
        self.Lx = Lx
        self.Ly = Ly
        self.corridor_width = corridor_width
        self.corridor_start = Ly / 2 - corridor_width / 2
        self.corridor_end = Ly / 2 + corridor_width / 2
        self.num_left = num_left
        self.num_right = num_right
        self.particles = self.spawn_particles()
        self.social_radius_same_direction = corridor_width / 4  # Reduced radius for same direction
        self.social_radius_opposite_direction = corridor_width / 2  # Larger radius for opposite directions
        self.wall_repelling_radius = 0.5
        self.social_force_strength = 4.0  # Reduced social force between people
        self.wall_force_strength = 50.0  # Strong repelling force from walls
        self.shear_force_strength = 2000.0  # Shear force along the walls
        self.last_velocities = [np.array([0.0, 0.0]) for _ in self.particles]  # Track previous velocities

    def spawn_particles(self):
        """Spawn particles on the left and right within the corridor bounds."""
        particles = []
        for _ in range(self.num_left):
            particles.append([
                self.Lx / 6,  # Left side x-position
                np.random.uniform(self.corridor_start, self.corridor_end),  # Random y-position in corridor
                1  # Moving right
            ])
        for _ in range(self.num_right):
            particles.append([
                5 * self.Lx / 6,  # Right side x-position
                np.random.uniform(self.corridor_start, self.corridor_end),  # Random y-position in corridor
                -1  # Moving left
            ])
        return particles

    def compute_potential(self, x, y, direction):
        """Compute a potential field guiding the particle to its target side."""
        if self.corridor_start <= y <= self.corridor_end:
            return abs(x - (0 if direction == -1 else self.Lx))  # Target is 0 for left, Lx for right
        else:
            return np.inf  # High cost for moving outside the corridor

    def compute_social_force(self, particle):
        """Compute the repulsive social force due to nearby particles."""
        x, y, direction = particle
        force = np.array([0.0, 0.0])
        for other in self.particles:
            if other is particle:
                continue
            ox, oy, other_direction = other
            dx, dy = x - ox, y - oy
            distance = np.sqrt(dx**2 + dy**2)
            if other_direction == direction:
                radius = self.social_radius_same_direction
            else:
                radius = self.social_radius_opposite_direction
            if distance < radius and distance > 0:
                strength = self.social_force_strength / distance**2
                direction_vector = np.array([dx, dy]) / distance
                force += strength * direction_vector
        return force

    def compute_wall_force(self, particle):
        """Compute a repelling force and shear force from the walls of the corridor."""
        x, y, direction = particle
        force = np.array([0.0, 0.0])
        # Repelling force
        if y - self.corridor_start < self.wall_repelling_radius:
            force[1] += self.wall_force_strength / (y - self.corridor_start + 1e-6) ** 2
        if self.corridor_end - y < self.wall_repelling_radius:
            force[1] -= self.wall_force_strength / (self.corridor_end - y + 1e-6) ** 2
        # Shear force to assist sliding along walls
        if y - self.corridor_start < self.wall_repelling_radius:
            force[0] += self.shear_force_strength * direction  # Shear force towards the target direction
        if self.corridor_end - y < self.wall_repelling_radius:
            force[0] += self.shear_force_strength * direction
        return force

    def compute_gradient(self, particle):
        """Compute the gradient of the potential at the particle's position."""
        x, y, direction = particle
        epsilon = 1e-3

        # Numerical gradient
        dphi_dx = (self.compute_potential(x + epsilon, y, direction) - self.compute_potential(x - epsilon, y, direction)) / (2 * epsilon)
        dphi_dy = (self.compute_potential(x, y + epsilon, direction) - self.compute_potential(x, y - epsilon, direction)) / (2 * epsilon)
        return np.array([dphi_dx, dphi_dy])

    def move_particle(self, particle, index, step_size=0.1):
        """Move the particle along the gradient-descent direction combined with social forces and wall repelling forces."""
        gradient = self.compute_gradient(particle)
        social_force = self.compute_social_force(particle)
        wall_force = self.compute_wall_force(particle)

        # Combine the forces
        total_force = -gradient + social_force + wall_force
        force_norm = np.linalg.norm(total_force)

        if force_norm > 0:
            direction = total_force / force_norm

            # Save the current position before moving
            previous_position = np.array(particle[:2])

            # Attempt to move the particle
            particle[:2] += step_size * direction

            # Ensure the particle stays within the corridor bounds
            if not (self.corridor_start <= particle[1] <= self.corridor_end):
                particle[:2] = previous_position  # Revert to the previous position if outside the corridor

            self.last_velocities[index] = direction

        # Ensure the particle stays within the x-boundaries
        particle[0] = np.clip(particle[0], 0, self.Lx)

    def plot_room(self, ax):
        """Plot the room layout and the particles' positions."""
        ax.clear()

        # Draw room boundaries
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        ax.set_aspect('equal')

        # Draw no-flux zones as rectangles
        no_flux_above = patches.Rectangle((self.Lx / 3, self.corridor_end), self.Lx / 3, self.Ly - self.corridor_end, 
                                          edgecolor='black', facecolor='gray', lw=1)
        no_flux_below = patches.Rectangle((self.Lx / 3, 0), self.Lx / 3, self.corridor_start, 
                                          edgecolor='black', facecolor='gray', lw=1)
        ax.add_patch(no_flux_above)
        ax.add_patch(no_flux_below)

        # Draw corridor
        corridor = patches.Rectangle((self.Lx / 3, self.corridor_start), self.Lx / 3, self.corridor_width, 
                                      edgecolor='none', facecolor='white', lw=1)
        ax.add_patch(corridor)

        # Add border around the entire plot
        border = patches.Rectangle((0, 0), self.Lx, self.Ly, edgecolor='black', facecolor='none', lw=2)
        ax.add_patch(border)

        # Draw the particles
        for particle in self.particles:
            ax.plot(particle[0], particle[1], 'ro' if particle[2] == 1 else 'bo', label=f'Particle {"Right" if particle[2] == 1 else "Left"}')

        # Add labels and title
        ax.set_title("Room Layout with Narrow Corridor (Particle Simulation)")
        ax.axis('off')
        ax.legend()

    def run_simulation(self, steps=100, step_size=0.1):
        """Run the particle simulation and create an animation."""
        fig, ax = plt.subplots(figsize=(10, 5))

        def update(frame):
            for i, particle in enumerate(self.particles):
                self.move_particle(particle, i, step_size)
            self.plot_room(ax)

        ani = FuncAnimation(fig, update, frames=steps, interval=100)
        plt.show()

# Create and run the simulation
simulation = ParticleSimulation(num_left=20, num_right=20)  # Adjust the number of particles here
simulation.run_simulation(steps=50, step_size=0.2)
