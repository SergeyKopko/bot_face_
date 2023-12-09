import argparse
import logging
import os
import time
import io
import torch
import uuid

import cv2
import insightface
import numpy as np
import onnxruntime
import datetime
import concurrent.futures

from constants import *
from swapper.face_swapper import Inswapper, paste_to_whole
from swapper.face_analyser import (
    detect_conditions,
    get_analysed_data,
    swap_options_list,
    analyse_face,
    get_single_face,
)
from swapper.face_enhancer import (
    get_available_enhancer_names,
    load_face_enhancer_model,
    cv2_interpolations,
)
from swapper.face_parsing import (
    init_parsing_model,
    get_parsed_mask,
    mask_regions,
    mask_regions_to_list,
)
from swapper.swapper import swap_face_with_condition, swap_specific
from swapper.utils import (
    trim_video,
    StreamerThread,
    ProcessBar,
    open_directory,
    split_list_by_lengths,
    merge_img_sequence_from_ref,
    create_image_grid,
    add_logo_to_image,
)


def read_image(content: bytes) -> np.ndarray:
    """
    Image bytes to OpenCV image

    :param content: Image bytes
    :returns OpenCV image
    :raises TypeError: If content is not bytes
    :raises ValueError: If content does not represent an image
    """
    if not isinstance(content, bytes):
        raise TypeError(
            f"Expected 'content' to be bytes, received: {type(content)}"
        )
    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"Expected 'content' to be image bytes")
    return image


## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Face Swapper")
parser.add_argument(
    "--out_dir", help="Default Output directory", default=os.getcwd()
)
parser.add_argument(
    "--cuda", action="store_true", help="Enable cuda", default=False
)
parser.add_argument(
    "--colab", action="store_true", help="Enable colab mode", default=False
)
user_args = parser.parse_args()

## ------------------------------ DEFAULTS ------------------------------

USE_COLAB = None
USE_CUDA = False
BATCH_SIZE = 32
WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "best detection"
DETECT_SIZE = 640
DETECT_THRESH = 0.4
NUM_OF_SRC_SPECIFIC = 10
MASK_INCLUDE = [
    "Skin",
    "R-Eyebrow",
    "L-Eyebrow",
    "L-Eye",
    "R-Eye",
    "Nose",
    "Mouth",
    "L-Lip",
    "U-Lip",
]
MASK_SOFT_KERNEL = 17
MASK_SOFT_ITERATIONS = 100
MASK_BLUR_AMOUNT = 0.15
MASK_ERODE_AMOUNT = 0.01

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None
FACE_ENHANCER_LIST = ["NONE"]
FACE_ENHANCER_LIST.extend(get_available_enhancer_names())
FACE_ENHANCER_LIST.extend(cv2_interpolations)
FACE_ENCHANCER_MODEL_CONST = None

## ------------------------------ SET EXECUTION PROVIDER ------------------------------
# Note: For AMD,MAC or non CUDA users, change settings here

PROVIDER = ["CPUExecutionProvider"]

if USE_CUDA:
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("\n********** Running on CUDA **********\n")
        PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        USE_CUDA = False
        print("\n********** CUDA unavailable running on CPU **********\n")
else:
    USE_CUDA = False
    print("\n********** Running on CPU **********\n")

device = "cuda" if USE_CUDA else "cpu"
EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None


## ------------------------------ LOAD MODELS ------------------------------


def load_face_analyser_model(name="buffalo_l"):
    # logging.info("start load_face_analyser_model")
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name=name,
            root="./swapper/insightface/assets/pretrained_models/insightface",
            providers=PROVIDER,
        )
        FACE_ANALYSER.prepare(
            ctx_id=0,
            det_size=(DETECT_SIZE, DETECT_SIZE),
            det_thresh=DETECT_THRESH,
        )
    # logging.info("finish load_face_analyser_model")


def load_face_swapper_model(
    path="./swapper/assets/pretrained_models/inswapper_128.onnx",
):
    # logging.info("start load_face_swapper_model")
    global FACE_SWAPPER
    # path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    if FACE_SWAPPER is None:
        batch = int(BATCH_SIZE) if device == "cuda" else 1
        FACE_SWAPPER = Inswapper(
            model_file=path, batch_size=batch, providers=PROVIDER
        )
    # logging.info("finish load_face_swapper_model")


def load_face_parser_model(
    path="./swapper/assets/pretrained_models/79999_iter.pth",
):
    # logging.info("start load_face_parser_model")
    global FACE_PARSER
    # path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    if FACE_PARSER is None:
        FACE_PARSER = init_parsing_model(path, device=device)
    # logging.info("finish load_face_parser_model")


## ------------------------------ MAIN PROCESS ------------------------------


def face_swapping_process(
    input_type,
    image_path,  # путь до фото референса
    source_path,  # путь до фото клиента
    output_path,
    condition,
    age,
    distance,
    face_enhancer_name,
    enable_face_parser,
    mask_includes,
    mask_soft_iterations,
    blur_amount,
    erode_amount,
    face_scale,
    enable_laplacian_blend,
    crop_top,
    crop_bott,
    crop_left,
    crop_right,
    images_state,
    watermark,
    *specifics,
):
    # logging.info("start start_face_swapping")
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    global FACE_ENCHANCER_MODEL_CONST
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

    ## ------------------------------ PREPARE INPUTS & LOAD MODELS ------------------------------
    # logging.info("start load_face_analyser_model")
    load_face_analyser_model()
    # logging.info("finish load_face_analyser_model")

    # logging.info("start load_face_swapper_model")
    load_face_swapper_model()
    # logging.info("finish load_face_swapper_model")

    if face_enhancer_name != "NONE":
        if FACE_ENCHANCER_MODEL_CONST == None:
            FACE_ENHANCER = load_face_enhancer_model(
                name=face_enhancer_name, device=device
            )
            FACE_ENCHANCER_MODEL_CONST = FACE_ENHANCER
        else:
            FACE_ENHANCER = FACE_ENCHANCER_MODEL_CONST
    else:
        FACE_ENCHANCER_MODEL_CONST = None
        FACE_ENHANCER = None

    if enable_face_parser:
        load_face_parser_model()

    includes = mask_regions_to_list(mask_includes)
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]
    if crop_top > crop_bott:
        crop_top, crop_bott = crop_bott, crop_top
    if crop_left > crop_right:
        crop_left, crop_right = crop_right, crop_left
    crop_mask = (crop_top, 511 - crop_bott, crop_left, 511 - crop_right)

    def swap_process(image_sequence, output_path):
        ## ------------------------------ CONTENT CHECK ------------------------------

        ## ------------------------------ ANALYSE FACE ------------------------------

        if condition != "Specific Face":
            source_data = source_path, age
        else:
            source_data = ((sources, specifics), distance)
        (
            analysed_targets,
            analysed_sources,
            whole_frame_list,
            num_faces_per_frame,
        ) = get_analysed_data(
            FACE_ANALYSER,
            image_sequence,
            source_data,
            swap_condition=condition,
            detect_condition=DETECT_CONDITION,
            scale=face_scale,
        )

        ## ------------------------------ SWAP FUNC ------------------------------

        preds = []
        matrs = []
        count = 0
        global PREVIEW
        for batch_pred, batch_matr in FACE_SWAPPER.batch_forward(
            whole_frame_list, analysed_targets, analysed_sources
        ):
            preds.extend(batch_pred)
            matrs.extend(batch_matr)
            EMPTY_CACHE()
            count += 1

            if USE_CUDA:
                image_grid = create_image_grid(batch_pred, size=128)
                PREVIEW = image_grid[:, :, ::-1]

        ## ------------------------------ FACE ENHANCEMENT ------------------------------

        generated_len = len(preds)
        if face_enhancer_name != "NONE":
            for idx, pred in enumerate(preds):
                enhancer_model, enhancer_model_runner = FACE_ENHANCER
                pred = enhancer_model_runner(pred, enhancer_model)
                preds[idx] = cv2.resize(pred, (512, 512))
        EMPTY_CACHE()

        ## ------------------------------ FACE PARSING ------------------------------

        if enable_face_parser:
            masks = []
            count = 0
            for batch_mask in get_parsed_mask(
                FACE_PARSER,
                preds,
                classes=includes,
                device=device,
                batch_size=BATCH_SIZE,
                softness=int(mask_soft_iterations),
            ):
                masks.append(batch_mask)
                EMPTY_CACHE()
                count += 1

                if len(batch_mask) > 1:
                    image_grid = create_image_grid(batch_mask, size=128)
                    PREVIEW = image_grid[:, :, ::-1]
            masks = np.concatenate(masks, axis=0) if len(masks) >= 1 else masks
        else:
            masks = [None] * generated_len

        ## ------------------------------ SPLIT LIST ------------------------------

        split_preds = split_list_by_lengths(preds, num_faces_per_frame)
        del preds
        split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
        del matrs
        split_masks = split_list_by_lengths(masks, num_faces_per_frame)
        del masks

        ## ------------------------------ PASTE-BACK ------------------------------

        def post_process(
            frame_idx,
            frame_img,
            split_preds,
            split_matrs,
            split_masks,
            enable_laplacian_blend,
            crop_mask,
            blur_amount,
            erode_amount,
        ):
            whole_img_path = frame_img
            whole_img = read_image(whole_img_path)
            blend_method = "laplacian" if enable_laplacian_blend else "linear"
            for p, m, mask in zip(
                split_preds[frame_idx],
                split_matrs[frame_idx],
                split_masks[frame_idx],
            ):
                p = cv2.resize(p, (512, 512))
                mask = (
                    cv2.resize(mask, (512, 512)) if mask is not None else None
                )
                m /= 0.25
                whole_img = paste_to_whole(
                    p,
                    whole_img,
                    m,
                    mask=mask,
                    crop_mask=crop_mask,
                    blend_method=blend_method,
                    blur_amount=blur_amount,
                    erode_amount=erode_amount,
                )
            if watermark:
                whole_img = add_logo_to_image(whole_img.astype("uint8"))
            _, image_target_b = cv2.imencode(".jpg", whole_img)
            images_state.image_target_byte.append(image_target_b)
            output_file = f"{output_path}/{uuid.uuid4().hex}.jpg"
            images_state.read_file_path = output_file
            images_state.image_target_links.append(output_file)

        def concurrent_post_process(image_sequence, *args):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for idx, frame_img in enumerate(image_sequence):
                    future = executor.submit(
                        post_process, idx, frame_img, *args
                    )
                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()

        concurrent_post_process(
            image_sequence,
            split_preds,
            split_matrs,
            split_masks,
            enable_laplacian_blend,
            crop_mask,
            blur_amount,
            erode_amount,
        )

    ## ------------------------------ IMAGE ------------------------------

    if input_type == "Image":

        # OUTPUT_FILE = output_file

        # logging.info("start swap_process")
        if isinstance(image_path, list):
            swap_process(image_path, output_path)
        else:
            swap_process([image_path], output_path)
        # logging.info("finish swap_process")

        WORKSPACE = output_path

        # images_state.queue_row.append(first_request_in_queue)


def check_analyse_face(
    source_path,
):
    try:
        load_face_analyser_model()
        analysed_source = analyse_face(
            read_image(source_path),
            FACE_ANALYSER,
            return_single_face=True,
            detect_condition="best detection",
            scale=1,
        )
        if analysed_source:
            return True
        return False
    except Exception as e:
        logging.exception("Лицо пользователя не распознано нейросетью")
        return False
