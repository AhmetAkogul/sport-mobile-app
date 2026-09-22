"""Kaydedilmiş bitirme-13 pozlarından türetilmiş, sabit FPS önizlemesi."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from capture.alignment import file_sha256
from capture.record import _write_json
from pose3d.iskelet import REFERANS_ISKELET as SKELETON


def validate_row(row, index, size, model):
    if (type(row.get('frame_index')) is not int or row['frame_index'] != index
            or row.get('skeleton') != SKELETON.ad
            or row.get('joints') != list(SKELETON.eklemler)
            or row.get('coordinate_space') != 'ozgun'
            or row.get('image_size') != list(size) or row.get('model') != model):
        raise ValueError('Poz kimliği, sırası veya görüntü boyutu eşleşmiyor.')
    points = np.asarray(row['points_px'], dtype=float)
    confidence = np.asarray(row['confidence'], dtype=float)
    visible = np.asarray(row['visible'])
    n = len(SKELETON)
    if (points.shape != (n, 2) or confidence.shape != (n,)
            or visible.shape != (n,) or visible.dtype != np.bool_
            or not np.isfinite(points).all() or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))):
        raise ValueError('Geçersiz poz dizileri.')
    return points, confidence, visible


class PoseRenderer:
    def __init__(self):
        import supervision as sv
        self.sv = sv
        color = sv.Color.from_hex('#00D5FF')
        # Supervision bağlantı indeksleri 1 tabanlıdır.
        edges = [(SKELETON.indeks(a) + 1, SKELETON.indeks(b) + 1)
                 for a, b in SKELETON.baglantilar]
        self.edges = sv.EdgeAnnotator(color=color, thickness=2, edges=edges)
        self.vertices = sv.VertexAnnotator(color=color, radius=4)

    def draw(self, image, arrays):
        points, confidence, visible = arrays
        keypoints = self.sv.KeyPoints(xy=points.astype(np.float32)[None],
            keypoint_confidence=confidence.astype(np.float32)[None], visible=visible[None])
        scene = self.edges.annotate(image.copy(), keypoints)
        return self.vertices.annotate(scene, keypoints)


def render_video(video_path, poses_dir, output_dir):
    video = Path(video_path).resolve()
    poses = Path(poses_dir)
    summary = json.loads((poses / 'summary.json').read_text())
    digest = file_sha256(video)
    count = summary.get('frames')
    if (summary.get('status') != 'completed' or summary.get('input_sha256') != digest
            or type(count) is not int or count <= 0 or not summary.get('model')):
        raise ValueError('Tamamlanmış poz kaydı ile kaynak video eşleşmeli.')
    renderer = PoseRenderer()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', input=str(video), input_sha256=digest,
                  poses_sha256=file_sha256(poses / 'poses.jsonl'), frames=0,
                  model=summary['model'], supervision_version=renderer.sv.__version__,
                  timing='constant_fps_preview', audio=False, physical_validation=False,
                  source_stop_reason=summary.get('stop_reason'))
    cap = writer = None
    _write_json(output / 'summary.json', report)
    try:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video veya FPS geçersiz.')
        report['fps'] = fps
        with (poses / 'poses.jsonl').open() as rows:
            for index in range(count):
                line = rows.readline()
                if not line:
                    raise ValueError('Poz kaydı erken bitti.')
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Kaynak video karesi okunamadı.')
                size = (frame.shape[1], frame.shape[0])
                arrays = validate_row(json.loads(line), index, size, summary['model'])
                annotated = renderer.draw(frame, arrays)
                if writer is None:
                    writer = cv2.VideoWriter(str(output / 'overlay.avi'),
                                             cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
                    if not writer.isOpened():
                        raise ValueError('Çıktı videosu açılamadı.')
                    if not cv2.imwrite(str(output / 'preview.png'), annotated):
                        raise OSError('Önizleme yazılamadı.')
                writer.write(annotated)
                report['frames'] += 1
            if rows.readline():
                raise ValueError('Özette belirtilenden fazla poz satırı var.')
        writer.release()
        writer = None
        check = cv2.VideoCapture(str(output / 'overlay.avi'))
        decoded = 0
        try:
            while True:
                ok, frame = check.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError('Çıktı boyutu değişti.')
                decoded += 1
        finally:
            check.release()
        if decoded != count:
            raise ValueError('Çıktı kare sayısı doğrulanamadı.')
        if file_sha256(video) != digest or file_sha256(poses / 'poses.jsonl') != report['poses_sha256']:
            raise ValueError('İşlem sırasında kaynak değişti.')
        report.update(status='completed', decoded_frames=decoded,
                      output_sha256=file_sha256(output / 'overlay.avi'))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        _write_json(output / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--poses-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(render_video(args.video, args.poses_dir, args.output), indent=2))


if __name__ == '__main__':
    main()
