import unittest
from control.pedestrian_identity import PedestrianIdentityAssociation


def point(identity, x, y=0., age=2, z=0.):
    return dict(track_id=identity, position=[x, y, z], age_frames=age)


class TestPedestrianIdentity(unittest.TestCase):
    def test_short_gap_id_switch_retains_measured_person(self):
        association = PedestrianIdentityAssociation()
        first = association.update([point('image-1', 0)], 1.)[0]['track_id']
        self.assertEqual(first, association.update([point('image-9', .8, age=1)], 1.4)[0]['track_id'])

    def test_ambiguous_neighbors_do_not_reassociate(self):
        association = PedestrianIdentityAssociation()
        old = {r['track_id'] for r in association.update([point('a', 0), point('b', .6)], 1.)}
        result = association.update([point('new', .3)], 1.1)
        self.assertNotIn(result[0]['track_id'], old)

    def test_two_new_tracks_cannot_claim_one_old_identity(self):
        association = PedestrianIdentityAssociation()
        old = association.update([point('a', 0)], 1.)[0]['track_id']
        result = association.update([point('b', .1), point('c', -.1)], 1.1)
        self.assertTrue(all(r['track_id'] != old for r in result))

    def test_long_gap_does_not_confirm_old_person_exit(self):
        association = PedestrianIdentityAssociation()
        old = association.update([point('a', 0)], 1.)[0]['track_id']
        self.assertNotEqual(old, association.update([point('a', .1)], 3.)[0]['track_id'])

    def test_same_image_identity_cannot_jump_to_other_person(self):
        association = PedestrianIdentityAssociation()
        old = association.update([point('a', 0)], 1.)[0]['track_id']
        self.assertNotEqual(old, association.update([point('a', 5)], 1.1)[0]['track_id'])

    def test_new_single_frame_detection_is_not_bound(self):
        self.assertEqual([], PedestrianIdentityAssociation().update([point('a', 0, age=1)], 1.))

    def test_other_road_level_not_associated(self):
        association = PedestrianIdentityAssociation()
        old = association.update([point('a', 0)], 1.)[0]['track_id']
        self.assertNotEqual(old, association.update([point('b', .1, z=1)], 1.1)[0]['track_id'])

    def test_absence_does_not_generate_an_observation(self):
        association = PedestrianIdentityAssociation()
        association.update([point('a', 0)], 1.)
        self.assertEqual([], association.update([], 1.1))

    def test_valid_image_identity_survives_projected_footpoint_jitter(self):
        association = PedestrianIdentityAssociation()
        old = association.update([point('a', 0)], 1.)[0]['track_id']
        self.assertEqual(old, association.update([point('a', .1, y=1.2)], 1.05)[0]['track_id'])

    def test_stable_image_identity_is_not_confused_by_nearby_person(self):
        association = PedestrianIdentityAssociation()
        old = association.update([point('a', 0), point('b', .6)], 1.)[0]['track_id']
        self.assertEqual(old, association.update([point('a', .1)], 1.1)[0]['track_id'])


if __name__ == '__main__':
    unittest.main()
