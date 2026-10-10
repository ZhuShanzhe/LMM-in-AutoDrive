from scenarios.complex.town05_scene2 import DeterministicSceneEvents


def test_nearby_cyclist_is_initialized_before_cold_fragment_capture():
    scene=DeterministicSceneEvents.__new__(DeterministicSceneEvents)
    event=dict(id='cycle',kind='cyclist',activate_progress_m=1500.,anchor_progress_m=1640.,complete_progress_m=1900.)
    scene.events=[event];scene.states={'cycle':'STAGED'}
    calls=[]
    scene._stage_cyclist=lambda event,progress:calls.append(('spawn',progress))
    scene._prestage_cyclist=lambda event,progress,**kwargs:calls.append(('place',progress,kwargs))
    scene._retire_event_actors=lambda *args,**kwargs:None
    assert scene.skip_events_before(1800.)==[]
    assert calls==[('spawn',1800.),('place',1800.,{'initial_setup':True})]
    assert scene.states['cycle']=='STAGED'


def test_future_and_skipped_cyclists_are_not_reintroduced():
    scene=DeterministicSceneEvents.__new__(DeterministicSceneEvents)
    scene.events=[dict(id='cycle',kind='cyclist',activate_progress_m=1500.,anchor_progress_m=1640.)]
    scene.states={'cycle':'STAGED'};calls=[]
    scene._stage_cyclist=lambda *args:calls.append('spawn')
    scene._prestage_cyclist=lambda *args,**kwargs:calls.append('place')
    scene._retire_event_actors=lambda *args,**kwargs:None
    assert scene.skip_events_before(1100.)==[] and calls==[]
    assert scene.skip_events_before(2400.)==['cycle'] and calls==[]
