from pathlib import Path
from unittest.mock import Mock

import pytest

from scene_understanding.realtime_perception import run, run_dataset
from scene_understanding.realtime_perception.cyclist_detection import CyclistAugmentedDetector
from scene_understanding.realtime_perception.detector import Detection


@pytest.mark.parametrize('module,source', [(run, '--capture-index'), (run_dataset, '--manifest')])
def test_generic_cli_accepts_bicycle_expert(monkeypatch, module, source):
    monkeypatch.setattr('sys.argv', ['perception', source, 'input.jsonl', '--output', 'out.jsonl',
        '--summary', 'summary.json', '--backend', 'yolop_yolo11', '--cyclist-weights', 'bike.pt'])
    assert module.parse_args().cyclist_weights == Path('bike.pt')


@pytest.mark.parametrize('module,source', [(run, '--capture-index'), (run_dataset, '--manifest')])
@pytest.mark.parametrize('backend', ['yolop', 'ssdlite'])
def test_expert_is_not_silently_ignored(monkeypatch, module, source, backend):
    monkeypatch.setattr('sys.argv', ['perception', source, 'input.jsonl', '--output', 'out.jsonl',
        '--summary', 'summary.json', '--backend', backend, '--cyclist-weights', 'bike.pt'])
    with pytest.raises(SystemExit):
        module.parse_args()


def test_dataset_backend_preserves_road_and_non_bicycle_categories(monkeypatch):
    from scene_understanding.realtime_perception import yolop_backend, ultralytics_backend
    road = Mock()
    road.latest_road_evidence = {'drivable_area': 'preserved'}
    road.detect.return_value = ([], 1.)
    car = Detection((0., 0., 10., 10.), .8, 'vehicle', 'car', 2)
    old_bike = Detection((20., 0., 30., 10.), .7, 'cyclist', 'bicycle', 1)
    new_bike = Detection((40., 0., 50., 20.), .9, 'cyclist', 'bicycle', 2)
    base, expert = Mock(), Mock()
    base.detect.return_value = ([car, old_bike], 2.)
    expert.detect.return_value = ([new_bike], 3.)
    monkeypatch.setattr(yolop_backend, 'YolopPanopticBackend', Mock(return_value=road))
    constructor = Mock(side_effect=[base, expert])
    monkeypatch.setattr(ultralytics_backend, 'UltralyticsTrafficDetector', constructor)
    monkeypatch.setattr('sys.argv', ['perception', '--manifest', 'input.jsonl', '--output', 'out.jsonl',
        '--summary', 'summary.json', '--cyclist-weights', 'bike.pt'])
    detector = run_dataset.build_detector(run_dataset.parse_args())
    detections, latency = detector.detect(object())
    assert [d.category for d in detections] == ['cyclist', 'vehicle']
    assert detections[0].class_id == 1
    assert detector.latest_road_evidence == road.latest_road_evidence
    assert latency == 6.
    assert constructor.call_args.kwargs['image_size'] == 640


@pytest.mark.parametrize('missing', ['base', 'expert'])
def test_batch_does_not_truncate_or_reassign_views(missing):
    base, expert = Mock(), Mock()
    base.detect_many.return_value = ([[], []], 1.)
    expert.detect_many.return_value = ([[], []], 2.)
    (base if missing == 'base' else expert).detect_many.return_value = ([[]], 1.)
    with pytest.raises(RuntimeError, match='every input view'):
        CyclistAugmentedDetector(base, expert).detect_many([object(), object()])


def test_batch_preserves_expert_class_identity_and_view_order():
    base, expert = Mock(), Mock()
    bike = Detection((1., 2., 3., 4.), .75, 'cyclist', 'bicycle', 2)
    base.detect_many.return_value = ([[], []], 1.)
    expert.detect_many.return_value = ([[bike], []], 2.)
    detections, latency = CyclistAugmentedDetector(base, expert).detect_many([object(), object()])
    assert len(detections[0]) == 1 and not detections[1]
    assert detections[0][0].class_id == 1 and latency == 3.
