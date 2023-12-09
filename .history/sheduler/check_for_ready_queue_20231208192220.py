import psycopg2
import logging
from constants import *
from t_commands.db import *
from swapper.app import check_analyse_face
from sheduler.utils import get_object_from_minio


def get_source_from_minio(image_ref_path):
    """Функция получает фото клиента для определение наличия лица."""

    # logging.info("start get_object_from_minio")
    try:
        object_data = MINIO_CLIENT.get_object(BUCKET_NAME, image_ref_path )
        return object_data.read()
    except:
        get_object_from_minio(image_ref_path, images_state=ImagesState)
        logging.exception(f"Ошибка извлечения {image_ref_path}")
    # logging.info("finish get_object_from_minio")


def check_for_ready_queue(db):
    """
    Функция реализует логику для выполнения задач
    из таблицы check_user_image_queue.
    """

    try:
        with psycopg2.connect(
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            database=env["DB_NAME"],
            host=env["DB_HOST"],
        ) as db:
            first_request_in_image_queue = get_first_in_check_user_image_queue(
                db
            )
            while first_request_in_image_queue:
                if (
                    first_request_in_image_queue[2] == USER_IMAGES_CHECK_PACK
                    or first_request_in_image_queue[2]
                    == USER_IMAGES_CHECK_PACK_LK
                ):
                    image_path_list = first_request_in_image_queue[4].split(
                        ";;"
                    )
                else:
                    image_path_list = [first_request_in_image_queue[4]]
                for i, image in enumerate(image_path_list):
                    is_find_face = check_analyse_face(
                        source_path=get_source_from_minio(
                            image
                        ),  # путь до фото клиента
                    )
                    if is_find_face:
                        change_find_face_status(
                            db,
                            first_request_in_image_queue[0],
                            READY,
                            image,
                            1,
                        )
                        for n in range(i + 1, len(image_path_list)):
                            MINIO_CLIENT.remove_object(
                                BUCKET_NAME, image_path_list[n]
                            )
                        first_request_in_image_queue = (
                            get_first_in_check_user_image_queue(db)
                        )
                        return
                    MINIO_CLIENT.remove_object(BUCKET_NAME, image)
                change_find_face_status(
                    db,
                    first_request_in_image_queue[0],
                    READY,
                    first_request_in_image_queue[4],
                    0,
                )
                first_request_in_image_queue = (
                    get_first_in_check_user_image_queue(db)
                )
                return
    except Exception:
        logging.exception("Возникла ошибка в check_for_ready_queue")
