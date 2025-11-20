import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from inference.models.sam3 import SegmentAnything3 as SAM3
from autodistill.helpers import load_image
import numpy as np

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SegmentAnything3(DetectionBaseModel):
    ontology: CaptionOntology
    
    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.model = SAM3()

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        results = self.model.segment_image(image, text=self.ontology.classes()[0])
        xyxy = sv.mask_to_xyxy(results[0])

        detections = sv.Detections(
            mask=results[0].astype(bool),
            confidence=results[1],
            xyxy=xyxy,
            class_id=np.array([0] * len(results[1])),
        )
        detections = detections[detections.confidence > confidence]

        return detections