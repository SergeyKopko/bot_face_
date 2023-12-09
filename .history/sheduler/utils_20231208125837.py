from constants import *
import logging
import io
from swapper.images_state import ImagesState


def get_object_from_minio(
    image_ref_path: str, images_state: ImagesState
) -> None:
    """ "Функция получает фото через minio."""

    # logging.info("start get_object_from_minio")
    try:
        object_data = MINIO_CLIENT.get_object(BUCKET_NAME, image_ref_path)
        print('get_object_from_minio', object_data)
        images_state.minio_image_list.append(object_data.read())
    except:
        get_object_from_minio(image_ref_path, images_state)
        logging.exception(f"Ошибка извлечения {image_ref_path}")
    # logging.info("finish get_object_from_minio")


def put_object_to_minio(image_target_link, i, images_state: ImagesState):
    """Функция загружает фото через minio."""

    # logging.info("start put_object_from_minio")
    try:
        with io.BytesIO(image_target_link) as data:
            MINIO_CLIENT.put_object(
                BUCKET_NAME,
                images_state.image_target_links[i],
                data,
                len(image_target_link),
            )
    except:
        put_object_to_minio(image_target_link, i, images_state)
        logging.exception(f"Ошибка извлечения {image_target_link}")
    # logging.info("finish put_object_from_minio")
