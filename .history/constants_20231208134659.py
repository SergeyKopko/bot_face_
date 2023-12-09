import os
from distutils.util import strtobool

from dotenv import load_dotenv
from minio import Minio
from pathlib import Path
import urllib3
import asyncpg
COUNT_PHOTO_FREE = 5
COUNT_PHOTO_TURBO = 10
COUNT_PHOTO_PREM = 100
COUNT_PHOTO_FOR_REFERAL = 5
REQUEST_TO_QUEUE_INTERVAL = 10
REQUEST_TO_QUEUE_FOR_READY_INTERVAL = 1
GET_FROM_MINIO_TASK_INTERVAL = 1
USER_IMAGE_CHECK = "image_check"
USER_IMAGES_CHECK_PACK = "images_check_pack"
USER_IMAGE_CHECK_LK = "image_check_lk"
USER_IMAGES_CHECK_PACK_LK = "images_check_pack_lk"
IMAGE_FREE = "image_free"
IMAGE_FOR_REFERAL = "image_for_referal"
IMAGE_PAID = "image_paid"
IMAGE_COLLAGE_5 = "image_collage_5"
IMAGE_COLLAGE_50 = "image_collage_50"
IMAGE_COLLAGE_100 = "image_collage_100"
PENDING = "pending"  # в ожидании обработки в очереди
DRAWING = "drawing"  # взято в работу нейросетью
READY = "ready"  # нейронка выполнила работу, необходимо отправить пользователю
BUCKET_NAME = "shawa"  # Minio bucket name

dotenv_path = Path(".env.dev")
load_dotenv(dotenv_path=dotenv_path)

env = os.environ

async def connect_to_db():
    db = await asyncpg.connect(
    user=env["POSTGRES_USER"],
    password=env["POSTGRES_PASSWORD"],
    database=env["DB_NAME"],
    host="127.0.0.1",
    )
    return db


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
MINIO_CLIENT = Minio(
    env["MINIO_SERVER"],
    access_key=env["MINIO_ROOT_USER"],
    secret_key=env["MINIO_ROOT_PASSWORD"],
    secure=strtobool(env["MINIO_SECURE"]),
)
MINIO_CLIENT._http = urllib3.PoolManager(num_pools=1000)

LOG_FORMAT = (
    '"%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(message)s"'
)
LOG_DATETIME_FORMAT = "%d.%m.%Y_%H:%M:%S"

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
