import numpy as np
import pytest
from control.crosswalk_conditions import observed_crosswalk_clearance


def inputs():
    polygon=np.array([[8.,-1.,0.],[12.,-1.,0.],[12.,1.,0.],[8.,1.,0.]])
    road=np.array([[x,0.,0.] for x in range(21)],dtype=float)
    ground=np.array([[x,y,0.] for x in np.linspace(8.05,11.95,20) for y in np.linspace(-.9,.9,8)])
    camera=np.eye(4);camera[2,3]=-2.
    return dict(polygons=[polygon],road=road,cloud=ground,inverse_camera=camera,
        image_size=(1280,1280),fov=100.,pedestrian_boxes=[])


@pytest.mark.parametrize('failure',[None,'no_region','ambiguous','object','person','no_cloud','partial',
    'different_height','behind','cropped','no_road'])
def test_crosswalk_clearance_requires_complete_region_evidence(failure):
    data=inputs()
    if failure=='no_region':data['polygons']=[]
    if failure=='ambiguous':data['polygons']*=2
    if failure=='object':data['cloud']=np.vstack([data['cloud'],[10.,0.,1.]])
    if failure=='person':data['pedestrian_boxes']=[[620.,620.,660.,800.]]
    if failure=='no_cloud':data['cloud']=None
    if failure=='partial':data['cloud']=data['cloud'][data['cloud'][:,0]<8.4]
    if failure=='different_height':data['road'][:,2]=4.
    if failure=='behind':data['inverse_camera'][0,3]=-20.
    if failure=='cropped':data['image_size']=(30,4)
    if failure=='no_road':data['road']=[]
    result=observed_crosswalk_clearance(**data)
    assert bool(result['satisfied'])==(failure is None)
    if failure is None:
        assert result['region_index']==0 and min(result['ground_support_counts'])>=3
