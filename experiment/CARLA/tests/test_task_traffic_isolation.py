from types import SimpleNamespace as NS
import pytest
from scenarios.task_traffic_isolation import prune_unrelated_task_corridor


def car(identity, x=15., y=0., z=0., yaw=0., role=None, destroyed=None):
    pose = NS(location=NS(x=x, y=y, z=z), rotation=NS(yaw=yaw))
    def destroy():
        if destroyed is not None:
            destroyed.append(identity)
        return True
    return NS(id=identity, is_alive=True, attributes={'role_name':role or f'background_{identity}'},
        get_transform=lambda:pose, destroy=destroy,
        bounding_box=NS(extent=NS(x=2., y=1.)))


def prune(values, **kwargs):
    return prune_unrelated_task_corridor([(0.,0.,0.,3.5), (30.,0.,0.,3.5)], values,
        owned_ids=kwargs.get('owned_ids', [v.id for v in values]),
        excluded_ids=kwargs.get('excluded_ids', []), protected_roles=kwargs.get('protected_roles', []))


def test_same_direction_background_including_adjacent_lane_is_isolated():
    assert [v['actor_id'] for v in prune([car(1),car(2,y=3.5)])] == [1,2]
    assert all(v['policy_input'] is False for v in prune([car(3)]))


@pytest.mark.parametrize('kwargs', [dict(x=-5.),dict(x=35.),dict(y=12.),dict(z=8.),dict(yaw=180.)])
def test_unrelated_geometry_is_preserved(kwargs):
    assert prune([car(1,**kwargs)]) == []


def test_protected_roles_ego_and_unowned_actors_are_never_removed():
    values=[car(1,role='hero'),car(2,role='ego'),car(3,role='task_lead'),car(4),car(5)]
    assert prune(values,owned_ids=[1,2,3,4],excluded_ids=[4],protected_roles=['task_lead']) == []


def test_curved_route_uses_local_heading_and_height():
    rows=[(0.,0.,0.,3.5),(15.,0.,0.,3.5),(15.,30.,0.,3.5)]
    result=prune_unrelated_task_corridor(rows,[car(1,x=15.,y=15.,yaw=90.),car(2,x=15.,y=15.,yaw=-90.)],
        owned_ids=[1,2],excluded_ids=[],protected_roles=[])
    assert [v['actor_id'] for v in result] == [1]


def test_failed_removal_is_not_reported_as_success():
    value=car(1);value.destroy=lambda:False
    with pytest.raises(RuntimeError):
        prune([value])


def test_invalid_route_is_rejected_before_deleting_anything():
    removed=[]
    with pytest.raises(ValueError):
        prune_unrelated_task_corridor([(0.,0.,0.,3.5),(30.,float('nan'),0.,3.5)],
            [car(1,destroyed=removed)],owned_ids=[1],excluded_ids=[],protected_roles=[])
    assert removed == []
