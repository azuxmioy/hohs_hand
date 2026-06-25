from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hohs_mano_regressor.data.artic_manifest import convert_artic_to_hamer_npz


class ArticManifestConversionTest(unittest.TestCase):
    def test_npz_arrays_convert_to_hamer_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "artic.npz"
            output = root / "hamer.npz"
            keypoints_2d = np.zeros((2, 21, 3), dtype=np.float32)
            keypoints_2d[:, :, 0] = np.linspace(10, 30, 21)
            keypoints_2d[:, :, 1] = np.linspace(20, 50, 21)
            keypoints_2d[:, :, 2] = 1.0
            keypoints_3d = np.ones((2, 21, 3), dtype=np.float32)
            np.savez(
                source,
                image_path=np.asarray(["images/a.jpg", "images/b.jpg"], dtype=object),
                hand_keypoints_2d=keypoints_2d,
                hand_keypoints_3d=keypoints_3d,
                bbox_xyxy=np.asarray([[0, 5, 40, 55], [1, 2, 41, 62]], dtype=np.float32),
                global_orient=np.zeros((2, 3), dtype=np.float32),
                mano_hand_pose=np.zeros((2, 45), dtype=np.float32),
                betas=np.zeros((2, 10), dtype=np.float32),
                hand_side=np.asarray(["right", "left"], dtype=object),
            )

            summary = convert_artic_to_hamer_npz(source, output)

            self.assertEqual(summary.samples, 2)
            with np.load(output, allow_pickle=True) as data:
                self.assertEqual(data["hand_keypoints_2d"].shape, (2, 21, 3))
                self.assertEqual(data["hand_keypoints_3d"].shape, (2, 21, 4))
                self.assertEqual(data["hand_pose"].shape, (2, 48))
                np.testing.assert_allclose(data["scale"][0], np.asarray([40.0, 50.0], dtype=np.float32))
                np.testing.assert_allclose(data["right"], np.asarray([1.0, 0.0], dtype=np.float32))

    def test_infers_bbox_from_visible_2d_keypoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "artic.npz"
            output = root / "hamer.npz"
            keypoints_2d = np.zeros((1, 21, 3), dtype=np.float32)
            keypoints_2d[0, 0] = [10, 20, 1]
            keypoints_2d[0, 1] = [30, 50, 1]
            keypoints_3d = np.zeros((1, 21, 4), dtype=np.float32)
            keypoints_3d[:, :, 3] = 1.0
            np.savez(
                source,
                image_path=np.asarray(["frame.jpg"], dtype=object),
                hand_keypoints_2d=keypoints_2d,
                hand_keypoints_3d=keypoints_3d,
            )

            convert_artic_to_hamer_npz(source, output, bbox_padding=1.5)

            with np.load(output, allow_pickle=True) as data:
                np.testing.assert_allclose(data["center"][0], np.asarray([20.0, 35.0], dtype=np.float32))
                np.testing.assert_allclose(data["scale"][0], np.asarray([30.0, 45.0], dtype=np.float32))
                self.assertEqual(float(data["has_hand_pose"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
