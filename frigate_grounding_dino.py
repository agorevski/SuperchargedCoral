#!/usr/bin/env python3

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from queue import Queue
import subprocess
from threading import local
import time
from pathlib import Path

import torch
from PIL import Image

import fiftyone as fo
import fiftyone.utils.coco as fouc

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


logger = logging.getLogger(__name__)
_worker_state = local()


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = "./frigate-dataset/"

DATASET_NAME = "frigate_grounding_dino7"

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
GROUNDING_DINO_DEVICE = os.getenv("GROUNDING_DINO_DEVICE", "auto").lower()
GROUNDING_DINO_BATCH_SIZE = int(os.getenv("GROUNDING_DINO_BATCH_SIZE", "8"))


def format_duration(seconds):

    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:

        return f"{hours}h {minutes}m {seconds}s"

    if minutes:

        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def chunked(items, size):

    if size < 1:

        raise ValueError("Batch size must be at least 1")

    for index in range(0, len(items), size):

        yield items[index:index + size]


# ============================================================
# Load model
# ============================================================

def load_model(device=None):

    if device is None:

        device = resolve_devices()[0]

    logger.info("Loading Grounding DINO on %s...", device)

    # Grounding DINO's deformable attention path can mix fp16 features with
    # fp32 sampling grids on CUDA, which fails inside torch.grid_sample.
    dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )

    model.to(device)
    model.eval()

    logger.info("Model loaded on %s with dtype %s", device, dtype)

    return processor, model


def resolve_devices():

    if GROUNDING_DINO_DEVICE == "auto":

        if torch.cuda.is_available():

            return [
                torch.device(f"cuda:{index}")
                for index in range(torch.cuda.device_count())
            ]

        return [torch.device("cpu")]

    if GROUNDING_DINO_DEVICE == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "GROUNDING_DINO_DEVICE=cuda was requested, but CUDA is not available"
        )

    if GROUNDING_DINO_DEVICE == "cuda":

        return [
            torch.device(f"cuda:{index}")
            for index in range(torch.cuda.device_count())
        ]

    if GROUNDING_DINO_DEVICE == "cpu":

        return [torch.device("cpu")]

    if GROUNDING_DINO_DEVICE.startswith("cuda:"):

        if not torch.cuda.is_available():

            raise RuntimeError(
                f"GROUNDING_DINO_DEVICE={GROUNDING_DINO_DEVICE} was requested, but CUDA is not available"
            )

        device = torch.device(GROUNDING_DINO_DEVICE)

        if device.index is not None and device.index >= torch.cuda.device_count():

            raise ValueError(
                f"GROUNDING_DINO_DEVICE={GROUNDING_DINO_DEVICE} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available"
            )

        return [device]

    raise ValueError(
        "GROUNDING_DINO_DEVICE must be one of: auto, cpu, cuda, cuda:<index>"
    )


def resolve_device():

    devices = resolve_devices()

    if len(devices) > 1:

        logger.info(
            "Multiple CUDA devices are available; using %s for single-model callers",
            devices[0],
        )

    return devices[0]



# ============================================================
# Create / load FiftyOne dataset
# ============================================================

def load_dataset():

    if DATASET_NAME in fo.list_datasets():

        logger.info(
            f"Loading existing dataset {DATASET_NAME}"
        )

        dataset = fo.load_dataset(
            DATASET_NAME
        )

    else:

        logger.info(
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


        logger.info(
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

        logger.info(
            f"FiftyOne will be available on your Tailnet at http://{tailnet_ip}:{FIFTYONE_PORT}"
        )

    else:

        logger.info(
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

    return infer_images(
        [image_path],
        processor,
        model,
    )[0]


def infer_images(
    image_paths,
    processor,
    model,
):

    images = []

    for image_path in image_paths:

        with Image.open(
            image_path
        ) as raw_image:

            image = raw_image.convert("RGB")
            image.load()
            images.append(image)

    inputs = processor(
        images=images,
        text=[PROMPT for _ in images],
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
            for image in images
        ],
    )

    return [
        detections_from_result(
            result,
            image.size,
        )
        for result, image in zip(
            results,
            images,
        )
    ]


def detections_from_result(
    result,
    image_size,
):

    detections = []

    width, height = image_size

    for score, label, box in zip(
        result["scores"],
        result["labels"],
        result["boxes"],
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


def initialize_inference_worker(model_contexts):

    device, processor, model = model_contexts.get_nowait()
    _worker_state.processor = processor
    _worker_state.model = model
    _worker_state.device = device


def infer_sample_batch_worker(sample_batch):

    detections = infer_images(
        [
            filepath
            for _, filepath in sample_batch
        ],
        _worker_state.processor,
        _worker_state.model,
    )

    return [
        (sample_id, detection)
        for (sample_id, _), detection in zip(
            sample_batch,
            detections,
        )
    ]


def run_batched_inference(
    samples_to_process,
    devices,
    started_at,
):

    if len(devices) > 1:

        return run_multi_gpu_batched_inference(
            samples_to_process,
            devices,
            started_at,
        )

    processor, model = load_model(devices[0])
    completed = 0
    pending = len(samples_to_process)

    for sample_batch in chunked(
        samples_to_process,
        GROUNDING_DINO_BATCH_SIZE,
    ):

        log_batch_started(
            completed,
            pending,
            sample_batch,
            devices[0],
        )

        detections_batch = infer_images(
            [
                sample.filepath
                for sample in sample_batch
            ],
            processor,
            model,
        )

        for sample, detections in zip(sample_batch, detections_batch):

            sample[LABEL_FIELD] = detections
            sample.save()
            completed += 1

        log_progress(
            completed,
            pending,
            sum(len(detections.detections) for detections in detections_batch),
            started_at,
        )

    return completed


def run_multi_gpu_batched_inference(
    samples_to_process,
    devices,
    started_at,
):

    pending = len(samples_to_process)
    completed = 0
    sample_by_id = {
        sample.id: sample
        for sample in samples_to_process
    }
    batches = [
        [
            (sample.id, sample.filepath)
            for sample in sample_batch
        ]
        for sample_batch in chunked(
            samples_to_process,
            GROUNDING_DINO_BATCH_SIZE,
        )
    ]
    logger.info(
        "Running inference with %s CUDA workers and batch size %s",
        len(devices),
        GROUNDING_DINO_BATCH_SIZE,
    )

    model_contexts = Queue()

    for device in devices:

        processor, model = load_model(device)
        model_contexts.put((device, processor, model))

    with ThreadPoolExecutor(
        max_workers=len(devices),
        initializer=initialize_inference_worker,
        initargs=(model_contexts,),
    ) as executor:

        futures = []

        for index, sample_batch in enumerate(batches):

            log_batch_started(
                index * GROUNDING_DINO_BATCH_SIZE,
                pending,
                [
                    sample_by_id[sample_id]
                    for sample_id, _ in sample_batch
                ],
                "CUDA pool",
            )
            futures.append(
                executor.submit(
                    infer_sample_batch_worker,
                    sample_batch,
                )
            )

        for future in as_completed(futures):

            detection_count = 0

            for sample_id, detections in future.result():

                sample = sample_by_id[sample_id]
                sample[LABEL_FIELD] = detections
                sample.save()
                completed += 1
                detection_count += len(detections.detections)

            log_progress(
                completed,
                pending,
                detection_count,
                started_at,
            )

    return completed


def log_batch_started(
    completed,
    pending,
    sample_batch,
    device,
):

    progress = completed + 1
    percent = (progress / pending) * 100 if pending else 100.0

    logger.info(
        "Running inference %s-%s/%s (%.1f%%) on %s: %s",
        progress,
        min(completed + len(sample_batch), pending),
        pending,
        percent,
        device,
        ", ".join(sample.filepath for sample in sample_batch),
    )


def log_progress(
    completed,
    pending,
    detection_count,
    started_at,
):

    elapsed = time.monotonic() - started_at
    images_per_second = completed / elapsed if elapsed else 0.0
    remaining = pending - completed
    eta = (
        remaining / images_per_second
        if images_per_second
        else 0.0
    )

    logger.info(
        "Finished inference %s/%s (%.1f%%, %s detections in last batch, %.2f img/s, elapsed %s, eta %s)",
        completed,
        pending,
        (completed / pending) * 100 if pending else 100.0,
        detection_count,
        images_per_second,
        format_duration(elapsed),
        format_duration(eta),
    )


def sample_is_processed(sample):

    if LABEL_FIELD not in sample:

        return False

    return sample[LABEL_FIELD] is not None



# ============================================================
# Main
# ============================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    dataset = load_dataset()


    logger.info(
        "Launching FiftyOne..."
    )


    session = launch_fiftyone(
        dataset
    )


    samples_to_process = [
        sample
        for sample in dataset
        if not sample_is_processed(sample)
    ]
    pending = len(samples_to_process)


    logger.info(
        f"{pending} images require inference"
    )


    started_at = time.monotonic()

    if pending:

        devices = resolve_devices()

        logger.info(
            "Using %s inference device(s) with batch size %s: %s",
            len(devices),
            GROUNDING_DINO_BATCH_SIZE,
            ", ".join(str(device) for device in devices),
        )

        run_batched_inference(
            samples_to_process,
            devices,
            started_at,
        )

    logger.info(
        "Inference complete"
    )

    logger.info(
        "FiftyOne session is still running; press Ctrl+C to stop the script"
    )

    session.wait()



if __name__ == "__main__":

    main()
