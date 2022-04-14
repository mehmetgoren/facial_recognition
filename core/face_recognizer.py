import importlib
from typing import List
from PIL import ImageDraw
import torch
import pickle

from common.utilities import logger, config
from core.utilities import base64_to_pil, pil_to_base64


class FaceCoor:
    crop_pil_image = None
    x1, y1, x2, y2 = 0, 0, 0, 0


class DetectedFace:
    pred_cls_idx: int = 0
    pred_cls_name: str = ''
    pred_score: float = .0
    crop_base64_image: str
    x1, y1, x2, y2 = 0, 0, 0, 0

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
        self.overlay = config.ai.read_service_overlay

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = self.facenet_pytorch.MTCNN(post_process=True, device=self.device)
        self.mtcnn.keep_all = True  # to detect all faces
        self.resnet = self.facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.resnet.classify = True

    def __prepare_pytorch_side(self, base64_img: str):
        aligned = []
        faces: List[FaceCoor] = []
        x = base64_to_pil(base64_img)
        with torch.no_grad():
            x_aligneds, probs = self.mtcnn(x, return_prob=True)
            if x_aligneds is not None:
                for j, x_aligned in enumerate(x_aligneds):
                    prob = probs[j]
                    if prob < self.mtcnn_threshold:
                        logger.info(f'mtcnn (recognizer) prob({prob}) is lower than the threshold({self.mtcnn_threshold})')
                        continue
                    aligned.append(x_aligned)
                    logger.info(f'Face detected with probability: {prob}')
                detected_faces, detected_faces_probs = self.mtcnn.detect(x)
                for j, detected_face in enumerate(detected_faces):
                    detected_faces_prob = detected_faces_probs[j]
                    if detected_faces_prob < self.mtcnn_threshold:
                        logger.info(f'detected face({prob}) threshold is under the minimum threshold({self.mtcnn_threshold})')
                        continue
                    x1, y1, x2, y2 = detected_face[0], detected_face[1], detected_face[2], detected_face[3]
                    face = x.crop((x1, y1, x2, y2))
                    fc = FaceCoor()
                    fc.crop_pil_image, fc.x1, fc.y1, fc.x2, fc.y2 = face, x1, y1, x2, y2
                    faces.append(fc)
            if len(aligned) == 0:
                logger.info('no face was detected')
                return x, faces, None
            aligned = torch.stack(aligned).to(self.device)
            embeddings = self.resnet(aligned)

        return x, faces, embeddings.detach().cpu()

    def __prepare_sklearn_side(self, face_images: List[FaceCoor], embeddings) -> List[DetectedFace]:
        svc = pickle.load(open('face_train_classifier_model.h5', 'rb'))
        svc.probability = True
        y_pred = svc.predict(embeddings)

        probas_all = svc.predict_proba(embeddings)

        class_names = pickle.load(open('class_names.h5', 'rb'))
        faces: List[DetectedFace] = []
        for index, p in enumerate(y_pred):
            prob = probas_all[index, p]
            class_name = class_names[p] if prob >= self.prob_threshold else 'unknown'
            fc: FaceCoor = face_images[index]
            df = DetectedFace()
            df.pred_score, df.pred_cls_idx, df.pred_cls_name = prob, p, class_name
            df.crop_base64_image = pil_to_base64(fc.crop_pil_image)
            df.x1, df.y1, df.x2, df.y2 = fc.x1, fc.y1, fc.x2, fc.y2
            faces.append(df)
        return faces

    def predict(self, base64_img: str) -> DetectedFaceImage:
        pil_image, face_images, embeddings = self.__prepare_pytorch_side(base64_img)
        if embeddings is None or len(face_images) == 0:
            return None
        detected_faces: List[DetectedFace] = self.__prepare_sklearn_side(face_images, embeddings)
        dfi = DetectedFaceImage()
        dfi.detected_faces = detected_faces
        if self.overlay:
            for detected_face in detected_faces:
                xy1 = (detected_face.x1, detected_face.y1)
                xy2 = (detected_face.x2, detected_face.y2)
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle((xy1, xy2), outline='yellow')
                text = detected_face.pred_cls_name
                draw.text(xy1, text)
            dfi.base64_image = pil_to_base64(pil_image)
        else:
            dfi.base64_image = base64_img
        return dfi
