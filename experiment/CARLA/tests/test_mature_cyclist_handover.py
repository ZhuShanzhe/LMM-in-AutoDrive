import unittest
from collections import deque
import numpy as np

from control.observed_vehicle_reidentification import bound_cyclist_handover_score
from control.passing_conditions import PassingMemory


def fixtures():
    road=np.column_stack([np.arange(-10., 201., 5.), np.zeros(43), np.zeros(43)])
    old=dict(category='cyclist', position=np.array([20.,0.]), velocity=np.array([3.,0.]),
        velocity_valid=False, last_seen_s=0., motion_history_established=True,
        validated_motion=dict(observed_at_s=0.,velocity=np.array([3.,0.])),
        appearances=deque([np.array([1.,0.])]), history=deque([(0.,np.array([20.,0.]))],maxlen=80),
        image_aliases={('front','old'):0.})
    history=deque([(9.+i*.05,np.array([47.+i*.15,0.])) for i in range(20)],maxlen=80)
    candidate=dict(category='cyclist', position=np.array([49.85,0.]), velocity=np.array([3.,0.]),
        velocity_valid=True, last_seen_s=9.95, motion_history_established=True,
        history=history, appearances=deque([np.array([1.,0.])]))
    m=dict(view='front',local_id='replacement',category='cyclist',position=np.array([50.,0.,1.]),
        surface=np.array([[49.9,-.1,1.],[49.9,.1,1.],[50.1,-.1,1.],[50.1,.1,1.]]),
        bbox_xyxy=[10.,10.,20.,20.],box_cropped=False,appearance=np.array([1.,0.]))
    return old,candidate,m,road


class MatureCyclistHandoverTests(unittest.TestCase):
    def test_mature_replacement_can_restore_bound_identity(self):
        old,candidate,m,road=fixtures()
        self.assertIsNotNone(bound_cyclist_handover_score(old,candidate,m,10.,road))

    def test_missing_motion_expiry_and_unsafe_matches_rejected(self):
        mutations=[('old', 'validated_motion', {}), ('old', 'category', 'vehicle'),
            ('candidate','velocity_valid',False), ('candidate','last_seen_s',9.),
            ('candidate','velocity',np.array([-3.,0.])),
            ('candidate','history',deque([(9.95,np.array([49.85,0.]))])),
            ('m','box_cropped',True), ('m','position',np.array([50.,3.,1.])),
            ('m','position',np.array([float('nan'),0.,1.])),
            ('m','appearance',np.array([.95,np.sqrt(1.-.95**2)])),
            ('m','appearance',np.array([0.,1.])), ('m','surface',np.zeros((2,3)))]
        for object_name,key,value in mutations:
            with self.subTest(object_name=object_name,key=key):
                old,candidate,m,road=fixtures()
                {'old':old,'candidate':candidate,'m':m}[object_name][key]=value
                self.assertIsNone(bound_cyclist_handover_score(old,candidate,m,10.,road))
        old,candidate,m,road=fixtures()
        candidate['last_seen_s']=31.
        self.assertIsNone(bound_cyclist_handover_score(old,candidate,m,31.,road))

    def make_memory(self):
        old,candidate,m,road=fixtures()
        memory=PassingMemory()
        memory.tracks={'bound':old,'replacement':candidate}
        memory.aliases={('front','replacement'):'replacement'}
        memory.bound={('request','pass'):{'target_id':'bound','bound_at_s':0.,
            'binding_origin':'measured_before_maneuver','clear_since_s':None,'last_frame':None,'last_time':None}}
        return memory,m,road

    def test_identity_consolidation_does_not_complete_passing(self):
        memory,m,road=self.make_memory()
        memory.update([m],frame=100,timestamp_s=10.,ego_xy=[45.,0.],heading=[1.,0.],road=road)
        self.assertNotIn('replacement',memory.tracks)
        self.assertEqual(memory.aliases[('front','replacement')],'bound')
        self.assertTrue(memory.tracks['bound']['velocity_valid'])
        self.assertEqual(memory.tracks['bound']['identity_repair_source'],'mature_measured_cyclist_handover')
        self.assertFalse(memory.facts)
        document=dict(request_id='request',intent=dict(steps=[dict(step_id='pass',action='PASS_BY',
            parameters=dict(condition='CYCLIST'))]))
        front=memory.proofs(document,frame_id='100',frame=100,timestamp_s=10.,observed_at_s=10.,
            ego_xy=[45.,0.],heading=[1.,0.],ego_rear_m=1.,corridor_clear=True,target_binding_allowed=False)
        self.assertFalse(front['pass']['satisfied'])
        for index in range(1,13):
            t=10.+index*.05
            measurement=dict(m,view='rear',local_id='rear',position=m['position']+np.array([index*.15,0.,0.]),
                surface=m['surface']+np.array([index*.15,0.,0.]))
            memory.update([measurement],frame=100+index,timestamp_s=t,ego_xy=[70.,0.],heading=[1.,0.],road=road)
            proof=memory.proofs(document,frame_id=str(100+index),frame=100+index,timestamp_s=t,
                observed_at_s=t,ego_xy=[70.,0.],heading=[1.,0.],ego_rear_m=1.,corridor_clear=True,
                target_binding_allowed=False)['pass']
            if index<11:self.assertFalse(proof['satisfied'])
        self.assertTrue(proof['satisfied'])
        self.assertGreaterEqual(proof['completed_event_fact']['measured_rear_clearance_m'],8.)

    def test_two_bound_candidates_cannot_be_arbitrarily_merged(self):
        memory,m,road=self.make_memory()
        extra,_,_,_=fixtures()
        memory.tracks['other-bound']=extra
        memory.bound[('request','other')]=dict(target_id='other-bound')
        memory.update([m],frame=100,timestamp_s=10.,ego_xy=[45.,0.],heading=[1.,0.],road=road)
        self.assertIn('replacement',memory.tracks)
        self.assertEqual(memory.aliases[('front','replacement')],'replacement')

    def test_two_fresh_replacements_with_matching_appearance_are_ambiguous(self):
        memory,m,road=self.make_memory()
        _,candidate,other_measurement,_=fixtures()
        candidate['position']=candidate['position']+np.array([1.,0.])
        candidate['history']=deque([(t,p+np.array([1.,0.])) for t,p in candidate['history']],maxlen=80)
        other_measurement=dict(other_measurement,position=other_measurement['position']+np.array([1.,0.,0.]),
            surface=other_measurement['surface']+np.array([1.,0.,0.]))
        candidate['measurements']=[other_measurement]
        memory.tracks['competing']=candidate
        memory.update([m],frame=100,timestamp_s=10.,ego_xy=[45.,0.],heading=[1.,0.],road=road)
        self.assertIn('replacement',memory.tracks)
        self.assertEqual(memory.aliases[('front','replacement')],'replacement')


if __name__=='__main__':unittest.main()
