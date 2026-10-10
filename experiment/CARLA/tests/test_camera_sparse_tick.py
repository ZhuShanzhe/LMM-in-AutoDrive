from types import SimpleNamespace as NS
from evaluation.camera import ExperimentCamera


def test_missing_video_frame_is_not_waited_for(tmp_path,monkeypatch):
    camera=ExperimentCamera(None,None,str(tmp_path),save_images=False)
    written=[]
    camera.video_writer=NS(write_raw=written.append)
    camera._capture_stride=2
    camera._on_image(NS(frame=101,raw_data=b'actual-frame'))
    assert camera.save_frame(101)
    original_get=camera._images.get
    def nonblocking_only(*,block=True,timeout=None):
        assert block is False and timeout is None, 'unsampled tick waited'
        return original_get(block=False)
    monkeypatch.setattr(camera._images,'get',nonblocking_only)
    assert not camera.save_frame(102)
    assert written==[b'actual-frame']


def test_sampled_frame_is_still_written_without_repeating_old_image(tmp_path):
    camera=ExperimentCamera(None,None,str(tmp_path),save_images=False)
    written=[]
    camera.video_writer=NS(write_raw=written.append)
    camera._capture_stride=2
    camera._on_image(NS(frame=11,raw_data=b'a'))
    assert camera.save_frame(11)
    assert not camera.save_frame(12)
    camera._on_image(NS(frame=13,raw_data=b'b'))
    assert camera.save_frame(13)
    assert written==[b'a',b'b']


def test_gpu_delayed_frames_are_written_with_their_own_overlay(tmp_path):
    camera=ExperimentCamera(None,None,str(tmp_path),save_images=False,video_overlay=True)
    written=[]
    camera.video_writer=NS(write_raw=written.append)
    camera._render_overlay=lambda raw,overlay:raw+str(overlay['frame']).encode()
    assert not camera.save_frame(21,{'frame':21})
    assert not camera.save_frame(22,{'frame':22})
    camera._on_image(NS(frame=21,raw_data=b'a'))
    assert camera.save_frame(23,{'frame':23})
    assert written==[b'a21']
    assert not camera.save_frame(24,{'frame':24})


def test_video_records_each_returned_frame_once_in_order(tmp_path):
    camera=ExperimentCamera(None,None,str(tmp_path),save_images=False)
    written=[]
    camera.video_writer=NS(write_raw=written.append)
    for frame in (5,3,3,7):
        camera._on_image(NS(frame=frame,raw_data=str(frame).encode()))
    assert camera.save_frame(5)
    assert written==[b'3',b'5']
    assert camera.save_frame(7)
    assert written==[b'3',b'5',b'7']
