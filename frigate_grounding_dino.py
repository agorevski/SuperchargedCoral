#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

import torch
from PIL import Image

import fiftyone as fo
import fiftyone.utils.coco as fouc

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = "./frigate-dataset/"

DATASET_NAME = "frigate_grounding_dino"

MODEL_ID = "IDEA-Research/grounding-dino-base"


# Objects you care about around your property
PROMPT = """
person.
dog.
cat.
deer.
bear.
bird.
raccoon.
vehicle.
car.
truck.
pickup truck.
amazon package.
delivery person.
bicycle.
motorcycle.
"""


CONFIDENCE_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

LABEL_FIELD = "grounding_dino"

FIFTYONE_ADDRESS = os.getenv("FIFTYONE_ADDRESS", "0.0.0.0")
FIFTYONE_PORT = int(os.getenv("FIFTYONE_PORT", "5151"))


# ============================================================
# Load model
# ============================================================

def load_model():

    print("Loading Grounding DINO...")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Grounding DINO's deformable attention path can mix fp16 features with
    # fp32 sampling grids on CUDA, which fails inside torch.grid_sample.
    dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    )

    model.to(device)
    model.eval()

    print("Model loaded")

    return processor, model



# ============================================================
# Create / load FiftyOne dataset
# ============================================================

def load_dataset():

    if DATASET_NAME in fo.list_datasets():

        print(
            f"Loading existing dataset {DATASET_NAME}"
        )

        dataset = fo.load_dataset(
            DATASET_NAME
        )

    else:

        print(
            "Creating dataset"
        )

        dataset = fo.Dataset(
            DATASET_NAME
        )

        dataset.persistent = True


        image_paths = list(
            Path(IMAGE_DIR)
            .rglob("*.[jJ][pP][gG]")
        )


        print(
            f"Adding {len(image_paths)} images"
        )


        for path in image_paths:

            sample = fo.Sample(
                filepath=str(path)
            )

            dataset.add_sample(sample)


    return dataset


def get_tailnet_ip():

    try:

        return subprocess.check_output(
            ["tailscale", "ip", "-4"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]

    except (FileNotFoundError, IndexError, subprocess.CalledProcessError):

        return None


def launch_fiftyone(dataset):

    tailnet_ip = get_tailnet_ip()

    if tailnet_ip:

        print(
            f"FiftyOne will be available on your Tailnet at http://{tailnet_ip}:{FIFTYONE_PORT}"
        )

    else:

        print(
            f"FiftyOne will listen on {FIFTYONE_ADDRESS}:{FIFTYONE_PORT}"
        )

    return fo.launch_app(
        dataset,
        address=FIFTYONE_ADDRESS,
        port=FIFTYONE_PORT,
        remote=True,
    )



# ============================================================
# Run Grounding DINO
# ============================================================

def infer_image(
    image_path,
    processor,
    model
):

    image = Image.open(
        image_path
    ).convert("RGB")


    inputs = processor(
        images=image,
        text=PROMPT,
        return_tensors="pt",
    )


    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    inputs = {
        key: value.to(
            device=model_device,
            dtype=model_dtype if torch.is_floating_point(value) else None,
        )
        for key, value in inputs.items()
    }


    with torch.no_grad():

        outputs = model(
            **inputs
        )


    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=CONFIDENCE_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[
            image.size[::-1]
        ],
    )[0]


    detections = []


    width, height = image.size


    for score, label, box in zip(
        results["scores"],
        results["labels"],
        results["boxes"],
    ):

        x1, y1, x2, y2 = box.tolist()


        detections.append(
            fo.Detection(
                label=label,
                confidence=float(score),
                bounding_box=[
                    x1 / width,
                    y1 / height,
                    (x2-x1) / width,
                    (y2-y1) / height,
                ],
            )
        )


    return fo.Detections(
        detections=detections
    )



# ============================================================
# Main
# ============================================================

def main():

    dataset = load_dataset()


    print(
        "Launching FiftyOne..."
    )


    session = launch_fiftyone(
        dataset
    )


    processor, model = load_model()


    pending = 0


    for sample in dataset:


        # Skip images already processed

        if LABEL_FIELD in sample:

            continue


        pending += 1


    print(
        f"{pending} images require inference"
    )


    completed = 0


    for sample in dataset:


        if LABEL_FIELD in sample:

            continue


        detections = infer_image(
            sample.filepath,
            processor,
            model
        )


        sample[LABEL_FIELD] = detections

        sample.save()


        completed += 1


        if completed % 25 == 0:

            print(
                f"Processed {completed}/{pending}"
            )


    print(
        "Inference complete"
    )


    session.wait()



if __name__ == "__main__":

    main()
