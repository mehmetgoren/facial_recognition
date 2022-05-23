import base64
import os
from io import BytesIO
from PIL import Image

from common.utilities import logger
from core.face_recognizer import FaceRecognizer
from core.face_trainer import FaceTrainer
from core.utilities import get_test_dir_path


def train():
    ft = FaceTrainer()
    ft.fit()


train()


def predict():
    test_dir_path = get_test_dir_path()
    base64_images = []
    root = os.path.join(test_dir_path, 't')
    for image_file in os.listdir(root):
        image = Image.open(os.path.join(root, image_file))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        base64_images.append(img_str.decode('utf-8'))

    fr = FaceRecognizer()
    index = -1
    for base64_image in base64_images:
        index += 1
        result = fr.predict(base64_image)
        if result is None:
            logger.info('image contains no face')
            continue
        for face in result.detected_faces:
            face_image = Image.open(BytesIO(base64.b64decode(face.crop_base64_image)))
            face_image.save(f'{test_dir_path}/{face.format()}.jpg')
        whole_image = Image.open(BytesIO(base64.b64decode(result.base64_image)))
        whole_image.save(f'{test_dir_path}/{str(index)}.jpg')


predict()
