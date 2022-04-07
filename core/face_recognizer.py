import base64
import importlib
from io import BytesIO
from typing import List
from PIL import Image
import torch
import pickle

from common.utilities import logger, config


class DetectedFace:
    pred_cls_idx: int = 0
    pred_cls_name: str = ''
    pred_score: float = .0
    crop_base64_image: str

    def format(self) -> str:
        return f'{self.pred_cls_idx}_{self.pred_cls_name}_{self.pred_score}'


class DetectedFaceImage:
    detected_faces: List[DetectedFace] = []
    base64_image: str = ''


class FaceRecognizer:
    def __init__(self):
        self.facenet_pytorch = importlib.import_module('facenet-pytorch')
        self.mtcnn_threshold = config.ai.face_recog_mtcnn_threshold
        self.prob_threshold = config.ai.face_recog_prob_threshold

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = self.facenet_pytorch.MTCNN(post_process=True, device=self.device)
        self.mtcnn.keep_all = True  # to detect all faces
        self.resnet = self.facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.resnet.classify = True

    @staticmethod
    def __base64_to_pil(base64_img: str):
        return Image.open(BytesIO(base64.b64decode(base64_img)))

    @staticmethod
    def __pil_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def __prepare_pytorch_side(self, base64_img: str):
        aligned = []
        faces = []
        x = self.__base64_to_pil(base64_img)
        with torch.no_grad():
            x_aligneds, probs = self.mtcnn(x, return_prob=True)
            if x_aligneds is not None:
                for j, x_aligned in enumerate(x_aligneds):
                    prob = probs[j]
                    if prob < self.mtcnn_threshold:
                        continue
                    aligned.append(x_aligned)
                    logger.info(f'Face detected with probability: {prob}')
                detected_faces, detected_faces_probs = self.mtcnn.detect(x)
                for j, detected_face in enumerate(detected_faces):
                    detected_faces_prob = detected_faces_probs[j]
                    if detected_faces_prob < self.mtcnn_threshold:
                        logger.info(f'detected face({prob}) threshold is under the minimum threshold({self.mtcnn_threshold})')
                        continue
                    logger.info(f'face: {detected_face}, {detected_faces_probs[j]}')
                    face = x.crop((detected_face[0], detected_face[1], detected_face[2], detected_face[3]))
                    faces.append(face)
            else:
                logger.info('no face was detected')
                return faces, None
            aligned = torch.stack(aligned).to(self.device)
            embeddings = self.resnet(aligned)

        return faces, embeddings.detach().cpu()

    def __prepare_sklearn_side(self, face_images: List[any], embeddings) -> List[DetectedFace]:
        svc = pickle.load(open('face_train_classifier_model.h5', 'rb'))
        svc.probability = True
        y_pred = svc.predict(embeddings)

        probas_all = svc.predict_proba(embeddings)

        class_names = pickle.load(open('class_names.h5', 'rb'))
        faces: List[DetectedFace] = []
        for index, p in enumerate(y_pred):
            prob = probas_all[index, p]
            class_name = class_names[p] if prob >= self.prob_threshold else 'other'
            face_image = face_images[index]
            df = DetectedFace()
            df.pred_score, df.pred_cls_idx, df.pred_cls_name = prob, p, class_name
            df.crop_base64_image = self.__pil_to_base64(face_image)
            faces.append(df)
        return faces

    def predict(self, base64_img: str) -> DetectedFaceImage:
        face_images, embeddings = self.__prepare_pytorch_side(base64_img)
        detected_faces: List[DetectedFace] = self.__prepare_sklearn_side(face_images, embeddings)
        dfi = DetectedFaceImage()
        dfi.detected_faces = detected_faces
        dfi.base64_image = base64_img
        return dfi
