"""Fine-tune a general road-user detector and retain independent bike metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import yaml
from ultralytics import YOLO
from scene_understanding.realtime_perception.taxonomy import map_detector_label


def evaluate_cyclists(weights,dataset,split):
    model=YOLO(str(weights))
    tp=fp=fn=negative_images=false_positive_images=0
    rows=[];times=[]
    images=sorted((dataset/'images'/split).glob('*.jpg'))
    for start in range(0,len(images),16):
        batch=images[start:start+16]
        results=model.predict([str(p) for p in batch],imgsz=640,conf=.25,iou=.45,device=0,verbose=False)
        for path,result in zip(batch,results):
            height,width=result.orig_shape
            ground=[]
            label=dataset/'labels'/split/(path.stem+'.txt')
            for line in label.read_text().splitlines():
                cls,x,y,w,h=map(float,line.split())
                if int(cls)==2:ground.append([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
            predicted=[]
            for box,confidence,cls in zip(result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                mapped=map_detector_label(str(result.names[int(cls)]))
                if mapped and mapped[0]=='cyclist':predicted.append((box,confidence))
            candidates=[]
            for gi,a in enumerate(ground):
                a=np.asarray(a)
                for pi,(b,score) in enumerate(predicted):
                    b=np.asarray(b)
                    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                    union=(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-overlap
                    iou=overlap/max(union,1e-9)
                    if iou>=.5:candidates.append((iou,gi,pi))
            matched_gt=set();matched_pred=set()
            for iou,gi,pi in sorted(candidates,reverse=True):
                if gi not in matched_gt and pi not in matched_pred:
                    matched_gt.add(gi);matched_pred.add(pi)
            frame_tp=len(matched_gt);frame_fp=len(predicted)-frame_tp;frame_fn=len(ground)-frame_tp
            tp+=frame_tp;fp+=frame_fp;fn+=frame_fn
            if not ground:
                negative_images+=1;false_positive_images+=int(bool(predicted))
            rows.append(dict(image=str(path.relative_to(dataset)),gt=len(ground),predictions=predicted,
                tp=frame_tp,fp=frame_fp,fn=frame_fn))
            times.append(float(result.speed['inference']))
    return dict(weights=str(weights),sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        split=split,images=len(images),conf=.25,iou_match=.5,tp=tp,fp=fp,fn=fn,
        precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),negative_images=negative_images,
        false_positive_images=false_positive_images,
        negative_image_false_positive_rate=false_positive_images/max(1,negative_images),
        batch16_inference_ms_p95=float(np.percentile(times,95)),
        timing_scope='batched offline detector inference; not streaming or end-to-end latency',rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--coco',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    args.output.mkdir(parents=True)
    config=yaml.safe_load((args.dataset/'dataset.yaml').read_text())
    mix=args.output/'mixed_training';mix.mkdir()
    for split in ('train','val','test'):
        for kind in ('images','labels'):
            destination=mix/kind/split;destination.mkdir(parents=True)
            for path in (args.dataset/kind/split).iterdir():
                if path.suffix in {'.jpg','.txt'}:(destination/path.name).symlink_to(path.resolve())
    candidates=[p for p in sorted((args.replay/'images/train').glob('*.jpg'))
        if p.is_file() and (args.replay/'labels/train'/(p.stem+'.txt')).is_file()]
    random.Random(20261008).shuffle(candidates)
    replay=candidates[:600]
    assert len(replay)>=64,'No usable real-image replay'
    for path in replay:
        (mix/'images/train'/path.name).symlink_to(path.resolve())
        (mix/'labels/train'/(path.stem+'.txt')).symlink_to(
            (args.replay/'labels/train'/(path.stem+'.txt')).resolve())
    config['path']=str(mix)
    (mix/'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    assert all(r['label_source']=='same_frame_semantic_mask_with_projected_actor_roi' for r in manifest['records'])
    (args.output/'protocol.json').write_text(json.dumps(dict(seed=20261008,epochs=8,imgsz=640,batch=16,
        init_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),replay_images=[str(p) for p in replay],
        validation='disjoint Town05 collection locations',test='Town04 collection; untouched during fitting',
        scope='new fine-tuning holdout; prior model pretraining may have included these maps',
        policy_truth_access=False,synthetic_label_source='RGB-aligned semantic mask + actor projection'),indent=2))
    baseline={}
    for name,weights in [('coco',args.coco),('published_carla',args.initial)]:
        baseline[name]=evaluate_cyclists(weights,args.dataset,'test')
        (args.output/f'{name}_test.json').write_text(json.dumps(baseline[name],indent=2))
        print(json.dumps(dict(model=name,**{k:v for k,v in baseline[name].items() if k!='rows'})),flush=True)
    model=YOLO(str(args.initial))
    model.train(data=str(mix/'dataset.yaml'),epochs=8,imgsz=640,batch=16,device=0,
        workers=2,seed=20261008,deterministic=True,amp=True,lr0=.0005,lrf=.1,
        warmup_epochs=1,mosaic=.2,close_mosaic=2,cos_lr=True,
        project=str(args.output),name='training',plots=False,patience=8,verbose=False,save=True)
    candidate=args.output/'training/weights/best.pt'
    validation=evaluate_cyclists(candidate,args.dataset,'val')
    test=evaluate_cyclists(candidate,args.dataset,'test')
    (args.output/'fine_tuned_val.json').write_text(json.dumps(validation,indent=2))
    (args.output/'fine_tuned_test.json').write_text(json.dumps(test,indent=2))
    summary={name:{k:v for k,v in value.items() if k!='rows'} for name,value in baseline.items()}
    summary['fine_tuned']={k:v for k,v in test.items() if k!='rows'}
    summary['selection_status']='candidate_only_no_native_acceptance'
    (args.output/'comparison.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
