import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from inference.core.entities.requests.sam3 import Sam3Prompt
from inference.models.sam3 import SegmentAnything3 as SAM3

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

        results = self.model.segment_image(
            image,
            prompts=[
                Sam3Prompt(type="text", text=prompt)
                for prompt in self.ontology.classes()
            ],
            format="polygon",
        )
        all_detections = []

        for item in results.prompt_results:
            preds = item.predictions

            all_polygons_coords = []
            all_confidences = []

            for p in preds:
                for polygon_coords in p.masks:
                    all_polygons_coords.append(polygon_coords)
                    all_confidences.append(p.confidence)

            xyxys = np.stack(
                [
                    sv.polygon_to_xyxy(np.array(poly_coords, dtype=np.int32))
                    for poly_coords in all_polygons_coords
                ],
                axis=0,
            )

            individual_masks = []
            height, width, _ = image.shape
            resolution_wh = (width, height)

            for poly_coords in all_polygons_coords:
                polygon_np = np.array(poly_coords, dtype=np.int32)
                mask = sv.polygon_to_mask(polygon_np, resolution_wh=resolution_wh)
                individual_masks.append(mask.astype(bool))

            masks = np.stack(individual_masks, axis=0)

            detections = sv.Detections(
                xyxy=xyxys,
                confidence=np.array(all_confidences, dtype=np.float32),
                mask=masks,
                class_id=np.array([item.prompt_index] * len(all_confidences)),
            )

            all_detections.append(detections)

        return sv.Detections.merge(all_detections)
