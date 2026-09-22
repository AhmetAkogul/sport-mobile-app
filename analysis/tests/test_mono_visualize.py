import json

import cv2
import numpy as np
import pytest

from capture.alignment import file_sha256
from mono.visualize import PoseRenderer, render_video, validate_row
from pose3d.iskelet import REFERANS_ISKELET as S


def row():
    return dict(frame_index=0, skeleton=S.ad, joints=list(S.eklemler),
                coordinate_space='ozgun', image_size=[100, 100], model='test',
                points_px=[[10 + i * 5, 50] for i in range(13)],
                confidence=[1.] * 13, visible=[True] * 13)


@pytest.mark.parametrize('key,value', [('frame_index', 1), ('skeleton', 'coco'),
    ('joints', list(reversed(S.eklemler))), ('coordinate_space', 'normalized'),
    ('image_size', [200, 100]), ('model', 'other'), ('visible', [1] * 13),
    ('confidence', [2.] * 13), ('points_px', [[float('nan'), 1]] * 13)])
def test_rejects_mismatched_row(key, value):
    data = row()
    data[key] = value
    with pytest.raises(ValueError):
        validate_row(data, 0, (100, 100), 'test')


def test_visibility_and_source_preserved():
    pytest.importorskip('supervision')
    renderer = PoseRenderer()
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    data = row()
    data['visible'] = [False] * 13
    assert not renderer.draw(source, validate_row(data, 0, (100, 100), 'test')).any()
    data['visible'][0] = True
    result = renderer.draw(source, validate_row(data, 0, (100, 100), 'test'))
    assert result[50, 10].any()
    assert not result[50, 15:].any()  # No invisible vertex or connecting edge.
    assert not source.any()


def fixture_files(tmp_path):
    video = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100, 100))
    assert writer.isOpened()
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    poses = tmp_path / 'poses'
    poses.mkdir()
    summary = dict(status='completed', input_sha256=file_sha256(video), frames=1, model='test')
    (poses / 'summary.json').write_text(json.dumps(summary))
    (poses / 'poses.jsonl').write_text(json.dumps(row()) + '\n')
    return video, poses


def test_render_roundtrip(tmp_path):
    pytest.importorskip('supervision')
    video, poses = fixture_files(tmp_path)
    before = video.read_bytes()
    report = render_video(video, poses, tmp_path / 'out')
    assert report['status'] == 'completed'
    assert report['decoded_frames'] == 1
    assert video.read_bytes() == before
    with pytest.raises(FileExistsError):
        render_video(video, poses, tmp_path / 'out')


def test_hash_mismatch(tmp_path):
    video, poses = fixture_files(tmp_path)
    video.write_bytes(video.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='eşleşmeli'):
        render_video(video, poses, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_extra_pose_fails_report(tmp_path):
    pytest.importorskip('supervision')
    video, poses = fixture_files(tmp_path)
    with (poses / 'poses.jsonl').open('a') as f:
        f.write(json.dumps(row()) + '\n')
    with pytest.raises(ValueError, match='fazla'):
        render_video(video, poses, tmp_path / 'out')
    assert json.loads((tmp_path / 'out' / 'summary.json').read_text())['status'] == 'failed'
