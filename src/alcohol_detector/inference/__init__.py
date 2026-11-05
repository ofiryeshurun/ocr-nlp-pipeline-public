"""Inference: image folder -> OCR -> classifier -> results CSV, and scoring against a split."""

from alcohol_detector.inference.evaluation import ConfusionMatrixPlot, SplitScore, SplitScorer
from alcohol_detector.inference.images import ImageDirectory
from alcohol_detector.inference.pipeline import AlcoholImagePipeline, EngineFactory, ImagePrediction, paddle_engine
from alcohol_detector.inference.writers import EVIDENCE_SEPARATOR, DetailsCSV, ResultsCSV

__all__ = [
    "EVIDENCE_SEPARATOR",
    "AlcoholImagePipeline",
    "ConfusionMatrixPlot",
    "DetailsCSV",
    "EngineFactory",
    "ImageDirectory",
    "ImagePrediction",
    "ResultsCSV",
    "SplitScore",
    "SplitScorer",
    "paddle_engine",
]
