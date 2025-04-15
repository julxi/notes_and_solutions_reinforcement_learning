import unittest
import numpy as np
from race_track import RaceTrack, parse_ascii_track, WALL, START, TRACK, FINISH


class TestParseAsciiTrack(unittest.TestCase):
    def test_simple_track(self):
        ascii_track = """
####
#--#
#S-#
#-F#
####
"""
        expected = np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 3, 0],
                [0, 1, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )

        result = parse_ascii_track(ascii_track)
        np.testing.assert_array_equal(result, expected)

    def test_get_position_of_piece(self):
        # given
        ascii_track = """
#-#
-#-
"""
        race_track = RaceTrack(ascii_track)

        # when
        result = race_track.get_position_of_piece(WALL)

        # then
        expected = [(0, 1), (1, 0), (2, 1)]
        self.assertCountEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
