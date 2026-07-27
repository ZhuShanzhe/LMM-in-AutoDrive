import unittest

from map_utils import resolve_carla_map_name


class CarlaMapNameTests(unittest.TestCase):
    def setUp(self):
        self.maps = ["Town05_Opt", "Town10HD_Opt"]

    def test_keeps_server_native_short_name(self):
        self.assertEqual(
            resolve_carla_map_name("Town05_Opt", self.maps), "Town05_Opt"
        )

    def test_resolves_configured_asset_path_to_server_name(self):
        self.assertEqual(
            resolve_carla_map_name("Carla/Maps/Town05_Opt", self.maps),
            "Town05_Opt",
        )

    def test_keeps_unknown_map_for_carla_to_report(self):
        self.assertEqual(
            resolve_carla_map_name("Carla/Maps/Unknown", self.maps),
            "Carla/Maps/Unknown",
        )


if __name__ == "__main__":
    unittest.main()
