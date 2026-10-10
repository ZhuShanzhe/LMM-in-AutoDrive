from unittest.mock import Mock
import numpy as np
import pytest
from scene_understanding.realtime_perception.ultralytics_backend import UltralyticsTrafficDetector


def detector():
    value=UltralyticsTrafficDetector.__new__(UltralyticsTrafficDetector)
    value.device='cpu';value.image_size=640;value.score_threshold=.25;value.iou_threshold=.45
    value.infrastructure_tiles=False;value.model=Mock();value._decode=Mock(side_effect=lambda item:item)
    return value


def test_batch_preserves_view_order_and_thresholds():
    value=detector();value.model.predict.return_value=['front','left','right','rear']
    images=[np.full((360,640,3),i,dtype=np.uint8) for i in range(4)]
    outputs,ms=value.detect_many(images)
    assert outputs==['front','left','right','rear'] and ms>=0.
    options=value.model.predict.call_args.kwargs
    assert options['conf']==.25 and options['iou']==.45 and options['imgsz']==640
    for i,image in enumerate(options['source']):assert np.all(image==i)


def test_batch_rejects_missing_output_instead_of_reassigning_views():
    value=detector();value.model.predict.return_value=['front']
    with pytest.raises(RuntimeError):value.detect_many([np.zeros((4,4,3)) for _ in range(2)])


def test_tiled_detector_keeps_existing_single_image_path():
    value=detector();value.infrastructure_tiles=True
    value.detect=Mock(side_effect=[(['first'],2.),(['second'],3.)])
    assert value.detect_many([1,2])==([['first'],['second']],5.)
    value.model.predict.assert_not_called()
