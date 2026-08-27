from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.test_utils import CustomTestCase


class TestPrecomputedEmbeddingLength(CustomTestCase):
    def test_rejects_short_embedding_before_scheduler_admission(self):
        processor = object.__new__(BaseMultimodalProcessor)
        processor.IM_START_TOKEN_ID = 1
        processor.IM_END_TOKEN_ID = 2
        processor.IM_TOKEN_ID = 3

        with patch.object(
            processor,
            "build_input_ids",
            return_value=([3] * 4010, [(0, 4009)], [Modality.IMAGE]),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "expected 4010 tokens at offset 0, got 877",
            ):
                processor.get_mm_data(
                    [],
                    {Modality.IMAGE: torch.zeros((877, 4))},
                )

    def test_accepts_exact_embedding_length(self):
        processor = object.__new__(BaseMultimodalProcessor)
        processor.IM_START_TOKEN_ID = 1
        processor.IM_END_TOKEN_ID = 2
        processor.IM_TOKEN_ID = 3

        embeddings = torch.zeros((4, 2))
        with patch.object(
            processor,
            "build_input_ids",
            return_value=([3] * 4, [(0, 3)], [Modality.IMAGE]),
        ):
            output = processor.get_mm_data([], {Modality.IMAGE: embeddings})

        torch.testing.assert_close(
            output.mm_items[0].precomputed_embeddings,
            embeddings,
        )


if __name__ == "__main__":
    unittest.main()
