<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Segment Anything 3 Module

This repository contains the code supporting the Segment Anything 3 base model for use with [Autodistill](https://github.com/autodistill/autodistill).

Segment Anything 3 (SAM3), developed by Meta Research, is a state-of-the-art zero-shot image segmentation model. You can prompt SAM3 with image regions and open text prompts.

Autodistill currently supports using text prompts to auto-label images for use in fine-tuning smaller vision models, such as an RF-DETR object detection model.

The Autodistill SAM 3 package only works on a GPU.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SAM3 Autodistill documentation](https://autodistill.github.io/autodistill/base_models/sam3/).

## Installation

To use SAM3 with Autodistill, you need to install the following dependency:


```bash
pip3 install autodistill-sam3
```

## Quickstart

```python
from autodistill_sam3 import SegmentAnything3
from autodistill.detection import CaptionOntology
from autodistill.helpers import load_image

import supervision as sv

# define an ontology to map class names to our SAM 3 prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = SegmentAnything3(
    ontology=CaptionOntology(
        {
            "fruit": "fruit",
            "leaf": "leaf"
        }
    )
)

# run inference on a single iamge
detections = base_model.predict("image.jpg")

image = load_image("image.jpg", return_format="cv2")

# visualise results
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections,
    labels=[base_model.ontology.classes()[class_id] for class_id in detections.class_id]
)
sv.plot_image(annotated_frame)

# label a folder of images
base_model.label("./images_to_label", extension=".jpeg")
```

Here is an example output:

<img width="765" height="660" alt="fruit" src="https://github.com/user-attachments/assets/f1de15a5-f49d-4ad1-8187-784395561c6c" />

## License

SAM 3 is licensed under a custom SAM license. [See the SAM 3 license in the official facebookresearch/sam3 repository](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
