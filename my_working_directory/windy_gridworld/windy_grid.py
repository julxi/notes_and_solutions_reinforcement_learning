import numpy as np

# position (x,y)


class WindyGrid:

    def __init__(self, width, height, wind_speeds, start, end):
        self.width = width
        self.height = height
        self.wind_speeds = wind_speeds
        self.start = start
        self.end = end

    def snap_to_grid(self, position):
        return np.clip(position, [0, 0], [self.width - 1, self.height - 1])
    
    def print_episode(self, positions):
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # mark trajectory
        for pos in positions:
            x, y = pos
            grid[y][x] = '*'

        # mark start and end
        sx, sy = self.start
        ex, ey = self.end
        grid[sy][sx] = 'S'
        grid[ey][ex] = 'E'

        # print grid from top to bottom
        for row in reversed(grid):
            print(' '.join(row))

                # print wind speeds below the grid
        print('-' * (2 * self.width - 1))
        print(' '.join(str(ws) for ws in self.wind_speeds))

