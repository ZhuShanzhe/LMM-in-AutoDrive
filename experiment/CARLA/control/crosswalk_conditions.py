"""Region-bound crosswalk clearance from static geometry and current sensors."""
import math
import cv2
import numpy as np


def closed_regions(locations):
    result=[];current=[]
    for point in locations:
        value=np.array([point.x,point.y,point.z],dtype=float)
        if len(current)>=3 and np.linalg.norm(value-current[0])<.05:
            result.append(np.asarray(current));current=[]
        else:current.append(value)
    return result


def observed_crosswalk_clearance(polygons,road,cloud,inverse_camera,image_size,fov,pedestrian_boxes):
    unknown=dict(satisfied=False,can_proceed=False,status='UNKNOWN',reason='crosswalk_region_not_observed')
    route=np.asarray(road,dtype=float)
    if route.ndim!=2 or route.shape[1]!=3 or len(route)<2 or not np.isfinite(route).all():return unknown
    lengths=np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)
    progress=np.r_[0.,np.cumsum(lengths)]
    candidates=[]
    for index,polygon in enumerate(polygons):
        polygon=np.asarray(polygon,dtype=float)
        if polygon.ndim!=2 or polygon.shape[1]!=3 or len(polygon)<3 or not np.isfinite(polygon).all():continue
        contour=polygon[:,:2].astype(np.float32)
        if cv2.contourArea(contour)<.5:continue
        intersect=[i for i,point in enumerate(route) if 3.<progress[i]<=40.
            and abs(point[2]-np.mean(polygon[:,2]))<.5
            and cv2.pointPolygonTest(contour,tuple(map(float,point[:2])),True)>=-.25]
        if intersect:candidates.append((float(progress[intersect[0]]),index,polygon,intersect[0]))
    if not candidates:return unknown
    candidates.sort(key=lambda item:item[0])
    if len(candidates)>1 and abs(candidates[0][0]-candidates[1][0])<=1.:
        return dict(unknown,reason='crosswalk_region_ambiguous')
    distance,index,polygon,route_index=candidates[0]
    unknown=dict(unknown,region_index=index,region_polygon_xyz=polygon.tolist(),region_distance_m=distance)
    matrix=np.asarray(inverse_camera,dtype=float);width,height=image_size
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0.,0.,0.,1.],atol=1e-5)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-3)
            or not isinstance(fov,(int,float)) or not 10.<fov<170.):return unknown
    camera=polygon@matrix[:3,:3].T+matrix[:3,3]
    if np.min(camera[:,0])<=.5:return dict(unknown,reason='crosswalk_not_fully_in_front_view')
    focal=width/(2*math.tan(math.radians(fov)/2))
    pixels=np.c_[width/2+focal*camera[:,1]/camera[:,0],height/2-focal*camera[:,2]/camera[:,0]]
    if not ((pixels[:,0]>2)&(pixels[:,0]<width-2)&(pixels[:,1]>2)&(pixels[:,1]<height-2)).all():
        return dict(unknown,reason='crosswalk_not_fully_visible')
    xmin,ymin=pixels.min(axis=0);xmax,ymax=pixels.max(axis=0)
    if (xmax-xmin)*(ymax-ymin)<64.:return dict(unknown,reason='crosswalk_projection_too_small')
    for box in pedestrian_boxes:
        x1,y1,x2,y2=box
        if min(xmax,x2)>max(xmin,x1) and min(ymax,y2)>max(ymin,y1):
            return dict(unknown,status='OCCUPIED',reason='pedestrian_overlaps_crosswalk_view')
    points=np.asarray(cloud,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<12 or not np.isfinite(points).all():return unknown
    center=polygon.mean(axis=0)
    design=np.c_[polygon[:,:2]-center[:2],np.ones(len(polygon))]
    plane,_,rank,_=np.linalg.lstsq(design,polygon[:,2],rcond=None)
    if rank<3 or np.max(np.abs(design@plane-polygon[:,2]))>.15:return unknown
    if np.linalg.norm(plane[:2])>.3:return dict(unknown,reason='unsupported_crosswalk_grade')
    bounds=(points[:,:2]>=polygon[:,:2].min(axis=0)-.3).all(axis=1)
    bounds&=(points[:,:2]<=polygon[:,:2].max(axis=0)+.3).all(axis=1)
    subset=points[bounds]
    contour=polygon[:,:2].astype(np.float32)
    subset=subset[np.asarray([cv2.pointPolygonTest(contour,tuple(map(float,p[:2])),True)>=-.3
                            for p in subset],dtype=bool)]
    heights=subset[:,2]-((subset[:,:2]-center[:2])@plane[:2]+plane[2])
    if ((heights>.25)&(heights<3.8)).any():
        return dict(unknown,status='OCCUPIED',reason='lidar_crosswalk_occupied')
    ground=subset[np.abs(heights)<=.18]
    if len(ground)<12:return dict(unknown,reason='crosswalk_ground_support_insufficient')
    segment_index=min(route_index,len(route)-2)
    heading=route[segment_index+1,:2]-route[segment_index,:2];norm=np.linalg.norm(heading)
    if norm<1.:return unknown
    heading/=norm
    limits=(polygon[:,:2]-center[:2])@heading
    along=(ground[:,:2]-center[:2])@heading
    bins=np.floor(np.clip((along-limits.min())/max(.1,np.ptp(limits))*3.,0.,2.999)).astype(int)
    support=np.bincount(bins,minlength=3)
    if np.min(support)<3:return dict(unknown,reason='crosswalk_partial_lidar_coverage',ground_support_counts=support.tolist())
    lateral=np.array([-heading[1],heading[0]])
    sides=(polygon[:,:2]-center[:2])@lateral
    across=(ground[:,:2]-center[:2])@lateral
    columns=np.floor(np.clip((across-sides.min())/max(.1,np.ptp(sides))*3.,0.,2.999)).astype(int)
    grid=np.bincount(bins*3+columns,minlength=9).reshape(3,3)
    if np.min(grid)<2:return dict(unknown,reason='crosswalk_partial_lidar_coverage',ground_support_grid=grid.tolist())
    return dict(satisfied=True,can_proceed=True,status='CLEAR',reason='observed_crosswalk_region_clear',
        region_index=index,region_polygon_xyz=polygon.tolist(),region_distance_m=distance,
        ground_support_counts=support.tolist(),ground_support_grid=grid.tolist(),policy_truth_access=False,
        semantic_scope='fully_projected_crosswalk_with_current_rgb_and_lidar_coverage')
