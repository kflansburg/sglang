import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.gemma4_vision import Gemma4VisionPooler
from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_processor(pooling_kernel_size=2):
    processor = Gemma4SGLangProcessor.__new__(Gemma4SGLangProcessor)
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(pooling_kernel_size=pooling_kernel_size)
    )
    return processor


def make_image_item(offsets, position_ids):
    position_ids = torch.as_tensor(position_ids).clone()
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=torch.zeros((*position_ids.shape[:2], 4)),
        model_specific_data={"image_position_ids": position_ids},
    )


class TestGemma4ImageTokenCounts(CustomTestCase):
    def test_validation_count_matches_vision_pooler_mask(self):
        positions = torch.tensor(
            [
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [2, 0],
                    [3, 0],
                    [2, 1],
                    [3, 1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ]
            ]
        )
        pooler = Gemma4VisionPooler(SimpleNamespace(hidden_size=4))
        _, mask = pooler._avg_pool_by_positions(
            torch.zeros((1, 12, 4)), positions, length=3
        )
        self.assertEqual(mask.sum().item(), 2)

        item = make_image_item(offsets=[(10, 11)], position_ids=positions)
        make_processor()._validate_image_token_counts([item])

    def test_accepts_matching_pooled_positions_with_padding(self):
        item = make_image_item(
            offsets=[(10, 11)],
            position_ids=[
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [2, 0],
                    [3, 0],
                    [2, 1],
                    [3, 1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ]
            ],
        )

        make_processor()._validate_image_token_counts([item])

    def test_rejects_placeholder_and_embedding_count_mismatch(self):
        item = make_image_item(
            offsets=[(10, 13)],
            position_ids=[
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [2, 0],
                    [3, 0],
                    [2, 1],
                    [3, 1],
                    [4, 0],
                    [5, 0],
                    [4, 1],
                    [5, 1],
                ]
            ],
        )

        with self.assertRaisesRegex(
            ValueError, r"placeholders=4, embeddings=3, patches=12"
        ):
            make_processor()._validate_image_token_counts([item])

    def test_rejects_production_placeholder_and_embedding_count_mismatch(self):
        positions = [
            position
            for cell in range(428)
            for position in (
                [2 * cell, 0],
                [2 * cell + 1, 0],
                [2 * cell, 1],
                [2 * cell + 1, 1],
            )
        ]
        item = make_image_item(offsets=[(0, 511)], position_ids=[positions])

        with self.assertRaisesRegex(
            ValueError, r"placeholders=512, embeddings=428, patches=1712"
        ):
            make_processor()._validate_image_token_counts([item])

    def test_validates_each_image_instead_of_only_aggregate_counts(self):
        item = make_image_item(
            offsets=[(0, 1), (2, 2)],
            position_ids=[
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ],
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [2, 0],
                    [3, 0],
                    [2, 1],
                    [3, 1],
                ],
            ],
        )

        with self.assertRaisesRegex(
            ValueError, r"item=0, image=0, placeholders=2, embeddings=1"
        ):
            make_processor()._validate_image_token_counts([item])

    def test_rejects_missing_position_ids(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 0)],
            feature=torch.zeros((1, 1)),
        )

        with self.assertRaisesRegex(ValueError, "missing image_position_ids"):
            make_processor()._validate_image_token_counts([item])

    def test_accepts_list_wrapped_split_item(self):
        positions = torch.tensor(
            [
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                ]
            ]
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 0)],
            feature=[torch.zeros((1, 8, 4))],
            model_specific_data={"image_position_ids": [positions]},
        )

        make_processor()._validate_image_token_counts([item])


if __name__ == "__main__":
    unittest.main()
