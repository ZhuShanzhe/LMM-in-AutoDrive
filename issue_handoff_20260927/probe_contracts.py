"""Small synthetic contract reproducers; no CARLA service or model weights."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'experiment/CARLA')]


def probes():
    import json
    from lightweight_vla_adapter.tests.fixtures import integration_documents
    from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
    from scene_understanding.src.control_decision import build_control_decision
    from scene2_runtime_interface import _scene2_step_contract,build_scheduled_driving_intent
    result={}
    d,w,a,r=integration_documents(parser_action='TURN',direction='RIGHT')
    missing=build_control_decision(d,w,a,r)
    d['intent']['steps'][0]['parameters']['target_location']={'x':10.,'y':10.}
    supplied=build_control_decision(d,w,a,r)
    result['turn_target']={'without':missing,'with':supplied,'scope':'synthetic low-risk interface input; not observed online failure'}
    for condition in ('PATH_CLEAR','PEDESTRIAN_CLEAR','PASSENGERS_CLEAR'):
        d,w,a,r=integration_documents(parser_action='WAIT')
        d['intent']['steps'][0].update(parameters={'condition':condition},completion={'type':'ACTION_REACHED'})
        runtime=DrivingPlanRuntime()
        for i in range(15):
            frame=f'probe_{i}'
            w['frame_id']=r['frame_id']=frame
            runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=0.)
            runtime.advance(w,r)
        result[condition]={'state':runtime.state,'scope':'low risk only; NO explicit actor-clear evidence; waiting can be correct'}
    result['return_when_safe']=_scene2_step_contract('CHANGE_LANE:RETURN_WHEN_SAFE')
    commands=json.loads((ROOT/'experiment/CARLA/configs/scene_2_town05_runtime.json').read_text())['commands']
    result['schedule_mode']=[dict(command_id=c['id'],plan=build_scheduled_driving_intent(c,0,0.,0.)) for c in commands]
    return result


if __name__=='__main__':
    import json
    print(json.dumps(probes(),ensure_ascii=False,indent=2))
