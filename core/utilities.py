import base64
import os
import uuid
from enum import Enum
from io import BytesIO
from threading import Thread
from PIL import Image

from common.data.heartbeat_repository import HeartbeatRepository
from common.data.service_repository import ServiceRepository
from common.utilities import crate_redis_connection, RedisDb, config


def create_dir_if_not_exist(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def base64_to_pil(base64_img: str):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def generate_id() -> str:
    return str(uuid.uuid4().hex)


def start_thread(fn, args):
    th = Thread(target=fn, args=args)
    th.daemon = True
    th.start()


def register_detect_service(service_name: str, instance_name: str, description: str):
    connection_main = crate_redis_connection(RedisDb.MAIN)
    heartbeat = HeartbeatRepository(connection_main, service_name)
    heartbeat.start()
    service_repository = ServiceRepository(connection_main)
    service_repository.add(service_name, instance_name, description)
    return connection_main


def get_train_dir_path() -> str:
    return os.path.join(config.general.root_folder_path, 'fr', 'ml', 'train')


def get_test_dir_path() -> str:
    return os.path.join(config.general.root_folder_path, 'fr', 'ml', 'test')


class EventChannels(str, Enum):
    read_service = 'read_service'
    fr_train_request = 'fr_train_request'
    fr_service = 'fr_service'
    fr_train_response = 'fr_train_response'
    frtc = 'frtc'
