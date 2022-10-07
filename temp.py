import base64
import os
from io import BytesIO
from PIL import Image, ImageDraw

from common.utilities import logger
from core.face_recognizer import FaceRecognizer
from core.face_trainer import FaceTrainer
from core.utilities import get_test_dir_path


def train():
    ft = FaceTrainer()
    ft.fit()


train()


def base64_to_pil(base64_img: str):
    return Image.open(BytesIO(base64.b64decode(base64_img)))


def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def predict():
    test_dir_path = get_test_dir_path()
    base64_images = []
    root = os.path.join(test_dir_path, 't')
    if not os.path.exists(root):
        os.makedirs(root)
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
        pil_image = base64_to_pil(base64_image)
        detected_faces = fr.predict(pil_image)
        if len(detected_faces) == 0:
            logger.info('image contains no face')
            continue
        for face in detected_faces:
            crop_image = pil_image.crop((face.x1, face.y1, face.x2, face.y2))
            crop_base64_image = pil_to_base64(crop_image)
            face_image = Image.open(BytesIO(base64.b64decode(crop_base64_image)))
            face_image.save(f'{test_dir_path}/{face.format()}.jpg')
        for detected_face in detected_faces:
            xy1 = (detected_face.x1, detected_face.y1)
            xy2 = (detected_face.x2, detected_face.y2)
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle((xy1, xy2), outline='yellow')
            text = detected_face.pred_cls_name
            draw.text(xy1, text)
        base64_image = pil_to_base64(pil_image)
        whole_image = Image.open(BytesIO(base64.b64decode(base64_image)))
        whole_image.save(f'{test_dir_path}/{str(index)}.jpg')


predict()
