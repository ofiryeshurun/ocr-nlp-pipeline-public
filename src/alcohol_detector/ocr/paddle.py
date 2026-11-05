"""PaddleOCR 3.x backend."""

from __future__ import annotations

import os

import numpy as np

from alcohol_detector.ocr.base import OCREngine, TextBox


def _preload_torch() -> None:
    """Import torch before paddle, if torch is installed.

    On Windows, once paddle has loaded its DLLs, a later ``import torch`` fails
    (WinError 127 in torch's shm.dll). Loading torch first avoids it.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


class PaddleOCREngine(OCREngine):
    """PaddleOCR text detection + recognition on CPU.

    oneDNN is always disabled: PaddlePaddle 3.3.1 crashes on Windows CPU with it
    on.
    """

    def __init__(
        self,
        det_model: str | None = None,
        rec_model: str | None = None,
        use_textline_orientation: bool = True,
        det_limit_side_len: int | None = None,
        det_limit_type: str | None = None,
    ) -> None:
        """Args:
        det_model: Detection model name (None = library default, PP-OCRv6_medium_det in 3.7).
        rec_model: Recognition model name (None = library default for English).
        use_textline_orientation: Run PaddleOCR's text-line orientation classifier.
        det_limit_side_len: Detection input size limit in pixels (None = library default: native size).
        det_limit_type: "max" shrinks the long side to the limit; "min" enlarges the short side to it.
        """
        if (det_limit_side_len is None) != (det_limit_type is None):
            raise ValueError("Set det_limit_side_len and det_limit_type together.")
        if det_limit_type not in (None, "max", "min"):
            raise ValueError(f"det_limit_type must be 'max' or 'min', got {det_limit_type!r}")
        # Skips PaddleX's start-up connectivity check of the model hosts; models still download on first use.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        _preload_torch()
        from paddleocr import PaddleOCR  # imported lazily: heavy, and optional for tests

        self.det_model: str | None = det_model
        self.rec_model: str | None = rec_model
        self.use_textline_orientation: bool = use_textline_orientation
        self.det_limit_side_len: int | None = det_limit_side_len
        self.det_limit_type: str | None = det_limit_type
        kwargs: dict[str, object] = dict(
            device="cpu",
            enable_mkldnn=False,
            # Whole-page rotation and dewarping are stages for scanned documents, so both are off.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,  # PaddleOCR 3.x's name for use_angle_cls
        )
        if det_model:
            kwargs["text_detection_model_name"] = det_model
        if det_limit_side_len is not None:
            kwargs["text_det_limit_side_len"] = det_limit_side_len
            kwargs["text_det_limit_type"] = det_limit_type
        if rec_model:
            kwargs["text_recognition_model_name"] = rec_model
        else:
            kwargs["lang"] = "en"
        self._ocr = PaddleOCR(**kwargs)

    @staticmethod
    def make_engine_id(
        det_model: str | None,
        rec_model: str | None,
        use_textline_orientation: bool = True,
        det_limit_side_len: int | None = None,
        det_limit_type: str | None = None,
    ) -> str:
        """Engine identifier for a configuration, without loading any model (used to find cached results).

        The size limit is only included when set, so ids of native-size runs are unchanged.
        """
        eid = f"paddle3|det={det_model or 'default'}|rec={rec_model or 'default-en'}|tlo={int(use_textline_orientation)}"
        if det_limit_side_len is not None:
            eid += f"|detsize={det_limit_type}{det_limit_side_len}"
        return eid

    @property
    def engine_id(self) -> str:
        """Identifier built from the model names and options."""
        return self.make_engine_id(
            self.det_model, self.rec_model, self.use_textline_orientation, self.det_limit_side_len, self.det_limit_type
        )

    def read(self, image: np.ndarray) -> list[TextBox]:
        """Run detection + recognition on a BGR image."""
        result = self._ocr.predict(image)[0]
        boxes: list[TextBox] = []
        for text, score, poly in zip(result["rec_texts"], result["rec_scores"], result["rec_polys"]):
            if text and text.strip():
                boxes.append(TextBox(text=text, confidence=float(score), polygon=np.asarray(poly).tolist()))
        return boxes
