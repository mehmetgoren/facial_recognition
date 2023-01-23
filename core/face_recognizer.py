import importlib
from typing import List
from PIL import Image
import torch
import pickle

from common.utilities import logger, config


class FaceCoor:
    x1, y1, x2, y2 = 0, 0, 0, 0


class DetectedFace:
    pred_cls_idx: int = 0
    pred_cls_name: str = ''
    pred_score: float = .0
    x1, y1, x2, y2 = 0, 0, 0, 0

    def format(self) -> str:
        return f'{self.pred_cls_idx}_{self.pred_cls_name}_{self.pred_score}'


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

        self.svc = pickle.load(open('face_train_classifier_model.h5', 'rb'))
        self.svc.probability = True
        self.class_names = pickle.load(open('class_names.h5', 'rb'))

    def __prepare_pytorch_side(self, x: Image):
        aligned = []
        faces: List[FaceCoor] = []
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
                    fc = FaceCoor()
                    fc.x1, fc.y1, fc.x2, fc.y2 = x1, y1, x2, y2
                    faces.append(fc)
            if len(aligned) == 0:
                logger.info('no face was detected')
                return faces, None
            aligned = torch.stack(aligned).to(self.device)
            embeddings = self.resnet(aligned)

        return faces, embeddings.detach().cpu()

    def __prepare_sklearn_side(self, face_images: List[FaceCoor], embeddings) -> List[DetectedFace]:
        y_pred = self.svc.predict(embeddings)
        probas_all = self.svc.predict_proba(embeddings)
        faces: List[DetectedFace] = []
        for index, p in enumerate(y_pred):
            prob = probas_all[index, p]
            class_name = self.class_names[p] if prob >= self.prob_threshold else 'unknown'
            fc: FaceCoor = face_images[index]
            df = DetectedFace()
            df.pred_score, df.pred_cls_idx, df.pred_cls_name = prob, p, class_name
            df.x1, df.y1, df.x2, df.y2 = fc.x1, fc.y1, fc.x2, fc.y2
            faces.append(df)
        return faces

    def predict(self, pil_image: Image) -> List[DetectedFace]:
        face_images, embeddings = self.__prepare_pytorch_side(pil_image)
        if embeddings is None or len(face_images) == 0:
            return None
        return self.__prepare_sklearn_side(face_images, embeddings)

    def reload_sklearn(self):
        self.svc = pickle.load(open('face_train_classifier_model.h5', 'rb'))
        self.class_names = pickle.load(open('class_names.h5', 'rb'))
