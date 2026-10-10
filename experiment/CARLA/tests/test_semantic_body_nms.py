import importlib.util
from pathlib import Path
from types import SimpleNamespace

from control import passing_conditions as module


def det(box,confidence=.9,category='vehicle'):
    return SimpleNamespace(bbox_xyxy=box,confidence=confidence,category=category)


def test_duplicate_vehicle_subtypes_share_one_body():
    first=det([0,0,100,100]);second=det([1,1,101,101],.7)
    assert module.semantic_body_nms([second,first])==[first]


def test_adjacent_vehicles_remain_distinct():
    a=det([0,0,100,100]);b=det([90,0,190,100],.8)
    assert len(module.semantic_body_nms([a,b]))==2


def test_cyclist_not_erased_by_overlapping_vehicle():
    a=det([0,0,100,100]);b=det([0,0,100,100],.8,'cyclist')
    assert len(module.semantic_body_nms([a,b]))==2


def test_empty_and_irrelevant_categories():
    assert module.semantic_body_nms([])==[]
    assert module.semantic_body_nms([det([0,0,5,5],category='pedestrian')])==[]
