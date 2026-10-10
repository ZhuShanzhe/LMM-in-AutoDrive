import pytest

from control.map_signal_observer import measured_stop_line_passage
from lightweight_vla_adapter.src.traffic_control_contract import TrafficControlContract


@pytest.mark.parametrize('fault',[None,'stale','wrong_signal','lateral','reverse','not_passed'])
def test_signal_retirement_requires_observed_physical_crossing(fault):
    entry=dict(position=[0.,0.,0.],forward=[1.,0.,0.])
    previous=dict(signal_id='lamp',timestamp_s=1.,front=[-.2,0.,0.],remaining_m=.2,
                  lateral_offset_m=0.,heading_alignment=1.)
    front=[1.1,0.,0.]
    if fault=='stale':previous['timestamp_s']=0.
    if fault=='wrong_signal':previous['signal_id']='other'
    if fault=='lateral':front[1]=4.
    if fault=='reverse':previous['front']=[2.,0.,0.]
    if fault=='not_passed':front[0]=.5
    proof=measured_stop_line_passage(entry,previous,front,1.1,'lamp',10)
    assert (proof is not None)==(fault is None)
    if proof:
        contract=TrafficControlContract();contract.last_blocked='lamp'
        decision=dict(action='keep_lane',target_speed_kmh=20.)
        out,info=contract.apply(decision,dict(applicable=False,retired_signals=[proof]),
                                timestamp_s=1.1,ego_speed_mps=2.)
        assert out==decision and info['retired_signal_release']
