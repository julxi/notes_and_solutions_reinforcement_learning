import numpy as np
import textwrap


WALL = 0
START = 1
TRACK = 2
FINISH = 3

WALL_ASCII = "#"
START_ASCII = "S"
TRACK_ASCII = "-"
FINISH_ASCII = "F"

ascii_to_code = {
    WALL_ASCII: WALL,
    START_ASCII: START,
    TRACK_ASCII: TRACK,
    FINISH_ASCII: FINISH,
}

code_to_ascii = {
    WALL: WALL_ASCII,
    START: START_ASCII,
    TRACK: TRACK_ASCII,
    FINISH: FINISH_ASCII,
}


def parse_ascii_track(ascii_track: str) -> np.ndarray:
    """
    Converts a multiline ASCII track to a NumPy array.
    The bottom line of the ASCII becomes row index 0.
    """
    lines = [line.strip() for line in ascii_track.strip().splitlines()]
    lines = lines[::-1]  # Flip vertically so y=0 is bottom
    height = len(lines)
    width = max(len(line) for line in lines)

    grid = np.full((height, width), WALL, dtype=np.uint8)
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            grid[y, x] = ascii_to_code[char]
    return grid


def bresenham_line_points(start: np.ndarray, end: np.ndarray) -> list[np.ndarray]:
    """
    Returns all (x, y) integer points on the line between start and end
    (inclusive), using Bresenham's algorithm.
    """
    x0, y0 = start
    x1, y1 = end
    points = []

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        points.append(np.array([x, y], dtype=int))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


class RaceTrack:
    def __init__(self, ascii_track: str):
        self._track = parse_ascii_track(ascii_track)
        self.height, self.width = self._track.shape

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_piece(self, x: int, y: int) -> int:
        return self._track[y, x]

    def get_position_of_piece(self, piece: int):
        positions = np.argwhere(self._track == piece)
        return [tuple(pos[::-1]) for pos in positions]



    def print_ascii(self, states: list[tuple[int, int, int, int]]):
        """Prints the ASCII track with multiple positions and velocity directions."""
        arrow_map = {
            (0, 0): "o",
            (0, -1): "↓",
            (0, 1): "↑",
            (1, 0): "→",
            (-1, 0): "←",
            (1, -1): "↘",
            (-1, -1): "↙",
            (1, 1): "↗",
            (-1, 1): "↖",
        }

        # Create a copy of the grid as a character array
        char_grid = [
            [code_to_ascii.get(cell, "?") for cell in row] for row in self._track
        ]

        # Overlay each state on the grid
        for i, (x, y, vx, vy) in enumerate(states):
            if 0 <= x < self.width and 0 <= y < self.height:
                arrow = arrow_map.get((np.sign(vx), np.sign(vy)), "x")
                char_grid[y][x] = arrow

        # Print from top to bottom (flip vertically)
        for row in reversed(char_grid):
            print("".join(row))

