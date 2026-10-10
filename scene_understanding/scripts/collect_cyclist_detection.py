"""Offline CARLA RGB/instance labels; never supplies actor labels to a policy."""
import argparse
import json
import queue
import random
from pathlib import Path
import carla
import numpy as np
from PIL import Image
from scene_understanding.core.carla_bbox_projection import project_actor_bbox


def wait_frame(source,frame):
    for _ in range(40):
        image=source.get(timeout=10)
        if image.frame==frame:return image
    raise RuntimeError('Sensor synchronization failed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Refuse to overwrite data'
    args.output.mkdir(parents=True)
    rng=random.Random(20261008)
    client=carla.Client('localhost',2000);client.set_timeout(90)
    records=[]
    classes={'vehicle.bh.crossbike':2,'vehicle.diamondback.century':2,'vehicle.gazelle.omafiets':2,
        'vehicle.kawasaki.ninja':3,'walker.pedestrian.0001':1,'vehicle.audi.a2':0}
    weathers=[carla.WeatherParameters.ClearNoon,carla.WeatherParameters.WetCloudyNoon,
        carla.WeatherParameters.ClearSunset,carla.WeatherParameters.HardRainSunset]
    for split,town,locations in [('train','Town05',10),('val','Town05',3),('test','Town04',5)]:
        print(json.dumps(dict(status='loading_map',split=split,town=town)),flush=True)
        world=client.load_world(town+'_Opt')
        original=world.get_settings()
        settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05
        world.apply_settings(settings)
        blueprints=world.get_blueprint_library()
        eligible=[world.get_map().get_waypoint(p.location) for p in world.get_map().get_spawn_points()]
        eligible=[p for p in eligible if not p.is_junction and p.lane_width>=3.]
        random.Random(20261008+int(town[-2:])).shuffle(eligible)
        chosen=eligible[:locations] if split!='val' else eligible[10:10+locations]
        print(json.dumps(dict(status='map_ready',split=split,locations=len(chosen))),flush=True)
        try:
            for group,waypoint in enumerate(chosen):
                weather=weathers[group%len(weathers)];world.set_weather(weather)
                for model,category in classes.items():
                    bp=blueprints.find(model)
                    target=world.try_spawn_actor(bp,carla.Transform(
                        waypoint.transform.location+carla.Location(z=.15),waypoint.transform.rotation))
                    if target is None:continue
                    sensors=[]
                    try:
                        target.set_simulate_physics(False)
                        world.tick()
                        rgb_queue=queue.Queue();instance_queue=queue.Queue()
                        for sensor_type,source in [('sensor.camera.rgb',rgb_queue),
                                ('sensor.camera.instance_segmentation',instance_queue)]:
                            sensor_bp=blueprints.find(sensor_type)
                            sensor_bp.set_attribute('image_size_x','640')
                            sensor_bp.set_attribute('image_size_y','384')
                            sensor_bp.set_attribute('fov','90')
                            sensor=world.spawn_actor(sensor_bp,waypoint.transform)
                            sensor.listen(source.put);sensors.append(sensor)
                        for distance in (8.,15.,25.,35.,45.,55.):
                            previous=waypoint.previous(distance)
                            if not previous:continue
                            camera_pose=previous[0].transform
                            camera_pose.location.z+=1.7
                            camera_pose.rotation.pitch=-3.
                            for yaw_offset in (-15.,0.,15.):
                                pose=carla.Transform(waypoint.transform.location+carla.Location(z=.15),
                                    carla.Rotation(yaw=waypoint.transform.rotation.yaw+yaw_offset))
                                target.set_transform(pose)
                                for sensor in sensors:sensor.set_transform(camera_pose)
                                for _ in range(3):frame=world.tick()
                                rgb=wait_frame(rgb_queue,frame);instance=wait_frame(instance_queue,frame)
                                pixels=np.frombuffer(instance.raw_data,dtype=np.uint8).reshape(384,640,4)
                                projected=project_actor_bbox(target,sensors[0],image_width=640,image_height=384,fov_deg=90)
                                allowed={0:(14,),1:(12,),2:(13,19),3:(13,18)}[category]
                                mask=np.isin(pixels[:,:,2],allowed)
                                roi=np.zeros(mask.shape,dtype=bool)
                                if projected is not None:
                                    x1,y1,x2,y2=np.asarray(projected)*[640,384,640,384]
                                    roi[max(0,int(y1)-8):min(384,int(y2)+9),
                                        max(0,int(x1)-8):min(640,int(x2)+9)]=True
                                ys,xs=np.where(mask & roi)
                                stem=f'{split}_{group:02d}_{model.replace(".","_")}_{distance:g}_{yaw_offset:g}'
                                image_dir=args.output/'images'/split;label_dir=args.output/'labels'/split
                                image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
                                bgr=np.frombuffer(rgb.raw_data,dtype=np.uint8).reshape(384,640,4)
                                Image.fromarray(bgr[:,:,:3][:,:,::-1]).save(image_dir/f'{stem}.jpg',quality=90)
                                if distance==8. and yaw_offset==0. and len(xs)<12:
                                    np.save(args.output/'invalid_instance_mask.npy',pixels)
                                    raise RuntimeError(f'Near target is not visible: actor={target.id}, actual={target.get_transform()}, requested={pose}')
                                label='';box=None
                                if len(xs)>=12:
                                    x1,x2,y1,y2=int(xs.min()),int(xs.max()+1),int(ys.min()),int(ys.max()+1)
                                    box=[x1,y1,x2,y2]
                                    label=f'{category} {(x1+x2)/1280:.7f} {(y1+y2)/768:.7f} {(x2-x1)/640:.7f} {(y2-y1)/384:.7f}\n'
                                (label_dir/f'{stem}.txt').write_text(label)
                                records.append(dict(image=f'images/{split}/{stem}.jpg',split=split,
                                    town=town,location_group=group,blueprint=model,class_id=category,
                                    distance_m=distance,yaw_offset=yaw_offset,visible_pixels=len(xs),box_xyxy=box,
                                    frame=frame,target_actor_id=target.id,weather=str(weather),
                                    label_source='same_frame_semantic_mask_with_projected_actor_roi',
                                    projected_box=projected,policy_truth_access=False))
                    finally:
                        for sensor in sensors:sensor.stop();sensor.destroy()
                        target.destroy()
                (args.output/'manifest.json').write_text(json.dumps(dict(records=records,
                    scope='offline detector training and independent collection holdout',seed=20261008),indent=2))
                print(json.dumps(dict(split=split,group=group,images=len(records))),flush=True)
        finally:
            world.apply_settings(original)
    import yaml
    (args.output/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(args.output),train='images/train',
        val='images/val',test='images/test',names={0:'vehicle',1:'pedestrian',2:'cyclist',3:'motorcycle',
            4:'traffic_light',5:'traffic_sign',6:'road_barrier',7:'traffic_cone'}),sort_keys=False))
    print(json.dumps(dict(total=len(records),status='collection_complete')),flush=True)


if __name__=='__main__':main()
