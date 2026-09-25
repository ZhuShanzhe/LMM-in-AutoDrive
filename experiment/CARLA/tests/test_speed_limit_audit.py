from types import SimpleNamespace as NS

from benchmark.speed_limit_audit import capture_speed_signs


def world_with(*actors):
    return NS(get_actors=lambda: NS(filter=lambda pattern: list(actors)))


def sign(identity, inside):
    return NS(id=identity, type_id='traffic.speed_limit.90',
              get_transform=lambda: NS(location=NS(x=0,y=0,z=0), rotation=NS(yaw=0)),
              trigger_volume=NS(extent=NS(x=1,y=1,z=1), contains=lambda loc,pose: inside(loc.x)))


def test_trigger_revisits_are_not_merged_and_no_limit_is_imputed():
    route=[dict(x=x,y=0,z=0,distance_m=i*5) for i,x in enumerate([0,1,2,3,1,0])]
    result=capture_speed_signs(world_with(sign(3,lambda x:x==1)),route,NS)
    assert result['signs'][0]['route_trigger_ranges_m']==[[5,5],[20,20]]
    assert result['signs'][0]['sign_speed_kmh']==90
    assert 'no_speeding_metric_overrides' in result['limitations']


def test_missing_trigger_is_visible_not_silently_accepted():
    actor=sign(1,lambda x:False)
    del actor.trigger_volume
    result=capture_speed_signs(world_with(actor),[],NS)
    assert result['signs'][0]['capture_error']


def test_no_signs_do_not_establish_unlimited_speed():
    result=capture_speed_signs(world_with(),[],NS)
    assert result['sign_count']==0
    assert result['scope']=='sampled_map_sign_triggers_not_complete_speed_limit_profile'
