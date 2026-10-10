from scenarios.complex.town05_scene2 import DeterministicSceneEvents


def manager(event, state="STAGED"):
    scene = object.__new__(DeterministicSceneEvents)
    scene.events = [event]
    scene.states = {event["id"]: state}
    calls = []
    scene._prestage_slow_vehicle = lambda *args, **kwargs: calls.append((args, kwargs))
    scene._retire_event_actors = lambda *args, **kwargs: None
    return scene, calls


def test_cold_start_in_prestage_window_places_actor_before_capture():
    event = dict(id="slow", kind="slow_vehicle", activate_progress_m=380,
                 anchor_progress_m=400)
    scene, calls = manager(event)
    assert scene.skip_events_before(335) == []
    assert calls == [((event, 335), {"initial_setup": True})]
    assert scene.states["slow"] == "STAGED"


def test_far_future_actor_remains_hidden_until_ordinary_prestaging():
    scene, calls = manager(dict(id="slow", kind="slow_vehicle",
                               activate_progress_m=380, anchor_progress_m=400))
    assert scene.skip_events_before(250) == []
    assert calls == []


def test_skipped_old_actor_is_not_reintroduced():
    scene, calls = manager(dict(id="slow", kind="slow_vehicle",
                               activate_progress_m=380, anchor_progress_m=400))
    assert scene.skip_events_before(800) == ["slow"]
    assert calls == []


def test_active_actor_is_not_repositioned():
    scene, calls = manager(dict(id="slow", kind="slow_vehicle",
                               activate_progress_m=380, anchor_progress_m=400), "ACTIVE")
    assert scene.skip_events_before(335) == []
    assert calls == []
