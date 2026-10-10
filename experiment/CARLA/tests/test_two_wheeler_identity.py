import copy
import numpy as np
import pytest

from control.passing_conditions import PassingMemory


def body(x, category='cyclist', identity='bike', cropped=False):
    return dict(category=category, view='rear', local_id=identity,
                position=np.array([x, 0., 1.]), box_cropped=cropped,
                surface=np.array([[x, -.2, 1.], [x, .2, 1.],
                                  [x+.2, -.2, 1.], [x+.2, .2, 1.]]))


def advance(memory, frame, values, ego=0.):
    now=frame*.05
    memory.update(values, frame=frame, timestamp_s=now, ego_xy=[ego, 0.], heading=[1., 0.])
    document=dict(request_id='cycle', intent=dict(steps=[dict(step_id='pass', action='PASS_BY',
                  parameters=dict(condition='CYCLIST'))]))
    return memory.proofs(document, frame_id=f'f{frame}', frame=frame, timestamp_s=now,
        observed_at_s=now, ego_xy=[ego, 0.], heading=[1., 0.], ego_rear_m=2.4,
        corridor_clear=True)['pass']


def mature():
    memory=PassingMemory()
    for frame in range(1,21):
        advance(memory, frame, [body(20+frame*.1)])
    return memory


@pytest.mark.parametrize('category',['motorcycle','pedestrian'])
def test_measured_rider_subtype_keeps_confirmed_bicycle_and_completes(category):
    memory=mature()
    # A rider continuation requires established bicycle motion, not a walker.
    if category=='pedestrian':
        for frame in range(21,41):advance(memory,frame,[body(22+(frame-20)*.15)])
        start=41
    else:start=21
    identity=memory.bound[('cycle','pass')]['target_id']
    for frame in range(start,start+15):
        x=25+(frame-40)*.15 if category=='pedestrian' else 20+frame*.1
        proof=advance(memory, frame, [body(x, category, 'new-subtype')], ego=60.)
        assert proof['target_id']==identity
        assert proof['target_observation']['measured_in_current_frame']
        assert proof['target_observation']['observed_categories']==[category]
    assert proof['satisfied']
    assert memory.tracks[identity]['category']=='cyclist'


@pytest.mark.parametrize('category', ['motorcycle','pedestrian'])
def test_unknown_subtype_cannot_create_bicycle(category):
    memory=PassingMemory()
    for frame in range(1,30):
        proof=advance(memory, frame, [body(20+frame*.1, category)])
    assert not proof['satisfied']
    assert not memory.bound


@pytest.mark.parametrize('failure', ['stale','wrong_position','ambiguous','cropped'])
def test_subtype_cannot_bypass_freshness_identity_or_rear_proof(failure):
    memory=mature()
    identity=memory.bound[('cycle','pass')]['target_id']
    if failure=='ambiguous':
        competing=copy.deepcopy(memory.tracks[identity])
        competing['position']+=np.array([.1,0.])
        memory.tracks['other-bike']=competing
    frame=28 if failure=='stale' else 21
    for current in range(frame, frame+15):
        x=20+current*.1+(4 if failure=='wrong_position' else 0)
        proof=advance(memory,current,[body(x,'motorcycle','flipped',failure=='cropped')],ego=50.)
    assert not proof['satisfied']


@pytest.mark.parametrize('similarity,accepted',[(1.,True),(.8,False)])
def test_short_subtype_occlusion_requires_appearance(similarity,accepted):
    memory=mature();track=next(iter(memory.tracks.values()))
    track['appearances']=[np.array([1.,0.])]
    measurement=body(22.8,'motorcycle','new')
    measurement['appearance']=np.array([similarity,np.sqrt(1-similarity**2)])
    score=memory.surface_association(track,measurement,1.4,np.array([1.,0.]),np.array([0.,1.]))
    assert (score is not None)==accepted
