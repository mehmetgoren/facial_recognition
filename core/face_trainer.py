import importlib
import os.path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
from addict import Dict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle

from common.utilities import logger, config
from core.utilities import create_dir_if_not_exist, get_train_dir_path


def _collate_fn(x):
    return x[0]


class FaceTrainer:
    def __init__(self):
        self.facenet_pytorch = importlib.import_module('facenet-pytorch')
        self.workers = 4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mtcnn_threshold = config.ai.face_recog_mtcnn_threshold
        self.mtcnn = self.facenet_pytorch.MTCNN(post_process=True, device=self.device)
        self.mtcnn.keep_all = False  # to detect only one face on training
        self.resnet = self.facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.resnet.classify = True

        self.folder_path = get_train_dir_path()
        create_dir_if_not_exist(self.folder_path)
        self.dataset = datasets.ImageFolder(self.folder_path)
        self.dataset.idx_to_class = {i: c for c, i in self.dataset.class_to_idx.items()}

        self.loader = DataLoader(self.dataset, collate_fn=_collate_fn, num_workers=self.workers)

    def __prepare_pytorch_side(self):
        aligned = []
        names = []
        for x, y in self.loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                if prob < self.mtcnn_threshold:
                    logger.info(f'mtcnn (training) prob({prob}) is lower than the threshold({self.mtcnn_threshold})')
                    continue
                aligned.append(x_aligned)
                names.append(self.dataset.idx_to_class[y])
                logger.info(f'Face detected with probability: {prob}')
        aligned = torch.stack(aligned).to(self.device)
        embeddings = self.resnet(aligned).detach().cpu()

        key = 0
        dic = Dict()
        classes = []
        for name in names:
            if name in dic:
                classes.append(dic[name])
            else:
                dic[name] = key
                classes.append(key)
                key += 1

        X = embeddings
        y = np.array(classes)
        return dic, X, y

    @staticmethod
    def __prepare_sklearn_side_and_save(dic, X, y):
        model_name = 'face_train_classifier_model.h5'
        if not os.path.exists(model_name):
            svc = LogisticRegression(max_iter=400)  # SVC(kernel="linear", probability=True)
        else:
            svc = pickle.load(open('face_train_classifier_model.h5', 'rb'))
        svc.fit(X, y)

        y_pred = svc.predict(X)
        acc = (y_pred == y).sum() / len(y) * 100.

        logger.info(f'y:      {y}')
        logger.info(f'y_pred: {y_pred}')
        logger.info(f'acc:    {acc}')

        # lets evaluate the success rate.
        cm = confusion_matrix(y_pred, y)
        logger.info(cm)
        # save the model to disk
        pickle.dump(svc, open(model_name, 'wb'))

        class_names = {v: k for k, v in dic.to_dict().items()}
        pickle.dump(class_names, open('class_names.h5', 'wb'))

    def fit(self):
        dic, X, y = self.__prepare_pytorch_side()
        self.__prepare_sklearn_side_and_save(dic, X, y)
