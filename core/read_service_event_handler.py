import base64
import io
from PIL import Image, UnidentifiedImageError
import json

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import logger, config
from core.face_recognizer import FaceRecognizer
from core.utilities import start_thread, EventChannels


class FrTrainCompleteEventHandler(EventHandler):
    def __init__(self, fr: FaceRecognizer):
        self.fr = fr

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        self.fr.reload_sklearn()
        logger.info('face training logistic regression data has been reloaded')


class ReadServiceEventHandler(EventHandler):
    def __init__(self):
        self.prob_threshold: float = config.ai.face_recog_prob_threshold
        self.encoding = 'utf-8'
        self.fr = FaceRecognizer()
        self.publisher = EventBus(EventChannels.snapshot_out)

    def start_listen_train_complete_event(self):
        start_thread(self._listen_train_complete, args=[])

    def _listen_train_complete(self):
        eb = EventBus(EventChannels.frtc)
        eh = FrTrainCompleteEventHandler(self.fr)
        eb.subscribe_async(eh)

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        start_thread(self.__handle, [dic])

    # you can handle it and add dict to multiprocessing.Queue then execute on a parameterless function by Pool.run_async to provide a real multi-core support
    def __handle(self, dic: dict):
        data: bytes = dic['data']
        dic = json.loads(data.decode(self.encoding))
        name = dic['name']
        source_id = dic['source']
        base64_image = dic['img']
        # open it when you want to use snapshot_in
        # source_id = dic['source_id']
        # base64_image = dic['base64_image']
        ai_clip_enabled = dic['ai_clip_enabled']

        base64_decoded = base64.b64decode(base64_image)
        try:
            image = Image.open(io.BytesIO(base64_decoded))
        except UnidentifiedImageError as err:
            logger.error(f'an error occurred while creating a PIL image from base64 string, err: {err}')
            return

        results = self.fr.predict(image)
        if results is None or len(results) == 0:
            logger.info(f'image contains no face for camera: {name}')
            return
        detected_faces = []
        face_logs = []
        for face in results:
            prob = float(face.pred_score)
            if prob < self.prob_threshold:
                logger.info(f'prob ({prob}) is lower than threshold: {self.prob_threshold} for {face.pred_cls_name}')
                continue
            dic_box = {'x1': int(face.x1), 'y1': int(face.y1), 'x2': int(face.x2), 'y2': int(face.y2)}
            detected_faces.append({'pred_score': prob, 'pred_cls_idx': int(face.pred_cls_idx), 'pred_cls_name': face.pred_cls_name, 'box': dic_box})
            face_logs.append({'pred_cls_name': face.pred_cls_name, 'pred_score': face.pred_score}),
        if len(detected_faces) == 0:
            logger.info('no detected face prob score is higher than threshold, this event will not be published')
            return

        dic = {'name': name, 'source': source_id, 'img': base64_image, 'ai_clip_enabled': ai_clip_enabled, 'detections': detected_faces,
               'channel': EventChannels.fr_service, 'list_name': 'detected_faces'}
        event = json.dumps(dic)
        self.publisher.publish(event)
        logger.info(f'face: detected {json.dumps(face_logs)}')
