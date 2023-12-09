import logging
import threading
import uuid
import concurrent.futures

import psycopg2
from apscheduler.schedulers.blocking import BlockingScheduler
from main import queue_state

from constants import *
from swapper.app import face_swapping_process
from sheduler.check_for_ready_queue import check_for_ready_queue
from swapper.images_state import ImagesState

from t_commands.db import *
from t_commands.make_collage import make_collage, make_ref_collage
from sheduler.utils import get_object_from_minio, put_object_to_minio
from typing import Union

scheduler = BlockingScheduler()


def get_task_from_db_queue() -> Union[tuple, None]:
    """Функция получает задачу из бд таблицы queue."""

    try:
        with psycopg2.connect(
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["DB_NAME"],
            host=env["DB_HOST"],
        ) as db:
            first_request_in_queue = get_first_request_type_in_queue(db)
            return first_request_in_queue
            # while first_request_in_queue:
            #     start_swapping(first_request_in_queue, db)
            #     first_request_in_queue = get_first_request_type_in_queue(db)
    except Exception:
        logging.exception("Возникла ошибка в send_request_to_db_queue")


def get_from_minio_task():
    """Подзадача для получения задачи из БД и получения фото."""

    if not queue_state.get_from_minio_task.empty():
        return
    task = get_task_from_db_queue()
    print(task)
    if not task:
        return
    logging.info(f"start get_from_minio_task {task[0]} {task[2]} {task[3]}")
    try:
        images_path = task[6].split(";;")
        images_state = ImagesState()
        images_state.task = task
        images_state.minio_image_list = []
        while len(images_state.minio_image_list) != task[3]:
            images_state.minio_image_list = []
            chunk_size = 100
            for n in range(0, len(images_path), chunk_size):
                tasks = []
                
                for image_ref_path in images_path[n : n + chunk_size]:
                    print('image', image_ref_path)
                    

                    
                    
                    
                    tasks.append(
                        threading.Thread(
                            target=get_object_from_minio,
                            args=(image_ref_path, images_state),
                        )
                    )

                for t in tasks:
                    t.start()
                    
                for t in tasks:
                    t.join()
                    

        # извлекаем фото пользователя
        object_data = MINIO_CLIENT.get_object(BUCKET_NAME, task[11])
        images_state.minio_user_image = object_data.read()
        object_data.close()
        object_data.release_conn()
        queue_state.get_from_minio_task.put(images_state)
    except:
        logging.exception(
            f"Ошибка извлечения фото через minio {task[0]} {task[2]} {task[3]}"
        )
    logging.info(f"finish get_from_minio_task {task[0]} {task[2]} {task[3]}")


def start_swapping_task():
    """Подзадача для получения задачи из БД и получения фото."""

    if queue_state.get_from_minio_task.empty():
        return
    try:
        images_state: ImagesState = queue_state.get_from_minio_task.get()
        task = images_state.task
        logging.info(
            f"start start_swapping_task {task[0]} {task[2]} {task[3]}"
        )
        images_state.image_target_links = []
        images_state.image_target_byte = []
        watermark = True if task[2] == IMAGE_FREE else False
        async_execute(
            images_state,
            images_state.minio_image_list,
            watermark,
            images_state.minio_user_image,
        )
        logging.info(
            f"finish start_swapping_task {task[0]} {task[2]} {task[3]}"
        )
        queue_state.swap_task.put(images_state)
    except:
        logging.exception(
            f"Ошибка в start_swapping_task {task[0]} {task[2]} {task[3]}"
        )


def async_execute(
    images_state: ImagesState, minio_image_list, watermark, minio_user_image
):
    try:
        face_swapping_process(
            input_type="Image",
            image_path=minio_image_list,  # путь до фоток референсов
            source_path=minio_user_image,  # путь до фото клиента
            output_path=f"images/{images_state.task[1]}/ready",
            condition="Biggest",
            age=25.0,
            distance=0.6,
            face_enhancer_name="GFPGAN",
            enable_face_parser=True,
            mask_includes=[
                "R-Eyebrow",
                "L-Eyebrow",
                "L-Eye",
                "R-Eye",
                "Nose",
                "Mouth",
                "L-Lip",
                "U-Lip",
                "Skin",
            ],
            mask_soft_iterations=100.0,
            blur_amount=0.1,
            erode_amount=0.001,
            face_scale=1,
            enable_laplacian_blend=True,
            crop_top=0,
            crop_bott=511,
            crop_left=0,
            crop_right=511,
            images_state=images_state,
            watermark=watermark,
            *[],
        )
    except Exception:
        logging.exception(
            f"Возникла ошибка в async_execute {images_state.task[0]} {images_state.task[2]} {images_state.task[3]}"
        )


def put_to_minio_task():
    """
    Функция выполняет подзадачу для выгрузки фото через minio client.
    Также вносит необходимые изменения в бд.
    """
    if queue_state.swap_task.empty():
        return
    try:
        images_state: ImagesState = queue_state.swap_task.get()
        task = images_state.task
        logging.info(f"start put_to_minio_task {task[0]} {task[2]} {task[3]}")
        # Используем ThreadPoolExecutor для многопоточной записи
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    put_object_to_minio, image_target_link, i, images_state
                )
                for i, image_target_link in enumerate(
                    images_state.image_target_byte
                )
            ]
        concurrent.futures.wait(futures)
        # queue_state.put_to_minio_task.get()
        logging.info(f"finish put_to_minio_task {task[0]} {task[2]} {task[3]}")

        # Записываем изменения в бд
        with psycopg2.connect(
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["DB_NAME"],
            host=env["DB_HOST"],
        ) as db:
            image_refs_links_str = ";;".join(task[6])
            image_target_links_str = ";;".join(images_state.image_target_links)
            add_to_history_image_links(
                task[4],
                task[1],
                task[7],
                image_refs_links_str,
                image_target_links_str,
                task[11],
                db,
            )
            if task[2] in [IMAGE_FOR_REFERAL]:
                add_history_collage_links(
                    task[1],
                    task[2],
                    task[4],
                    db,
                )
            if task[2] in [IMAGE_FREE, IMAGE_FOR_REFERAL]:
                collage_link = make_ref_collage(
                    images_state.image_target_byte,
                    task[1],
                    task[4],
                    1187,
                    500,
                    db,
                )
                update_image_colage_link(task[4], collage_link, db)
            if task[2] in [IMAGE_COLLAGE_50, IMAGE_COLLAGE_100]:
                collage_link = make_collage(
                    images_state.image_target_byte,
                    task[1],
                    task[4],
                    1187,
                    200,
                )
                update_image_colage_link(task[4], collage_link, db)
            remove_requests_from_queue(task[0], task[4], db)
    except IndexError:
        if task[3] != task[7]:
            if task[3] % 10 == 0:
                target_image_links = remove_requests_from_queue(
                    task[0],
                    task[4],
                    db,
                )
            return
        target_image_links = remove_requests_from_queue(
            task[0],
            task[4],
            db,
        )
    except Exception:
        logging.exception(
            f"Возникла ошибка в start_swapping {task[0]} {task[2]} {task[3]}"
        )


# Функция реализует логику для выполнения задач
# из таблицы check_user_image_queue.
scheduler.add_job(
    check_for_ready_queue,
    trigger="interval",
    seconds=REQUEST_TO_QUEUE_FOR_READY_INTERVAL,
    kwargs={"db": False},
)

# Функция выполняет подзадачу для получения задачи из БД из таблицы queue
# и загрузки изображений через minio client
scheduler.add_job(
    get_from_minio_task,
    trigger="interval",
    seconds=GET_FROM_MINIO_TASK_INTERVAL,
    kwargs={},
)

# Функция выполняет подзадачу для получения задачи из БД из таблицы queue
# Также вносит необходимые изменения в бд.
scheduler.add_job(
    start_swapping_task,
    trigger="interval",
    seconds=GET_FROM_MINIO_TASK_INTERVAL,
    kwargs={},
)

# Функция выполняет подзадачу для выгрузки фото через minio client
scheduler.add_job(
    put_to_minio_task,
    trigger="interval",
    seconds=GET_FROM_MINIO_TASK_INTERVAL,
    kwargs={},
)
