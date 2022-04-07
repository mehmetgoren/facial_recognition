import base64
import os
from io import BytesIO
from PIL import Image

from common.utilities import config
from core.face_recognizer import FaceRecognizer
from core.face_trainer import FaceTrainer


def train():
    ft = FaceTrainer()
    ft.fit()


train()


def predict():
    base64_images = []
    root = os.path.join(config.ai.face_recog_folder, 'test', 't')
    for image_file in os.listdir(root):
        image = Image.open(os.path.join(root, image_file))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        base64_images.append(img_str.decode('utf-8'))

    img_dir = os.path.join(config.ai.face_recog_folder, 'test')
    fr = FaceRecognizer()
    for base64_image in base64_images:
        result = fr.predict(base64_image)
        # todo: publish it later.
        for face in result.detected_faces:
            face_image = Image.open(BytesIO(base64.b64decode(face.crop_base64_image)))
            face_image.save(f'{img_dir}/{face.format()}.jpg')


predict()
