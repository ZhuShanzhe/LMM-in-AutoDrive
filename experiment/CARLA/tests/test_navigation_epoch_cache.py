from types import SimpleNamespace as NS
from control.generic_route_pid import GenericRoutePID


def case():
    obj=object.__new__(GenericRoutePID);obj.route_plan=[];obj.route_manager=None
    obj._observation_epoch=100;obj._action='keep_lane';obj._completed_lane_navigation=True
    obj._navigation_branch_samples=None;obj.progress_m=lambda:12.
    calls=[]
    def query(distance,**kwargs):
        calls.append(distance);return NS(distance=distance)
    obj._uncached_current_lane_navigation_point=query
    return obj,calls,NS(x=1.,y=2.,z=.1)


def request(obj,pose):
    return obj._current_lane_navigation_point(10.,ego_location=pose,ego_yaw_deg=30.)


def test_same_epoch_pose_route_and_branch_reuses_identical_waypoint():
    obj,calls,pose=case();a=request(obj,pose);b=request(obj,pose)
    assert a is b and len(calls)==1


def test_new_physics_epoch_invalidates_cache():
    obj,calls,pose=case();request(obj,pose);obj._observation_epoch+=1;request(obj,pose)
    assert len(calls)==2


def test_changed_pose_invalidates_cache():
    obj,calls,pose=case();request(obj,pose);pose.x+=.001;request(obj,pose)
    assert len(calls)==2


def test_changed_branch_invalidates_cache():
    obj,calls,pose=case();request(obj,pose);obj._navigation_branch_samples=[];request(obj,pose)
    assert len(calls)==2


def test_mutating_branch_query_is_never_cached():
    obj,calls,pose=case()
    def query(distance,**kwargs):
        calls.append(distance);obj._navigation_branch_samples=[];return NS(distance=distance)
    obj._uncached_current_lane_navigation_point=query
    request(obj,pose);request(obj,pose)
    assert len(calls)==2


def test_unscoped_queries_retain_original_behavior():
    obj,calls,pose=case();obj._observation_epoch=None
    request(obj,pose);request(obj,pose);assert len(calls)==2
