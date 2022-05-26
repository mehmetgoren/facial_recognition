import json
import multiprocessing as mp

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import logger
from core.face_trainer import FaceTrainer
from core.utilities import EventChannels


class TrainEventHandler(EventHandler):
    def __init__(self):
        mp.set_start_method('spawn')

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        proc = mp.Process(target=_train)
        try:
            proc.start()
            proc.join()
        finally:
            proc.kill()
        logger.info('a training sub-process has been terminated')


def _train():
    result = True
    try:
        trainer = FaceTrainer()
        trainer.fit()
    except BaseException as ex:
        logger.error(f'an error occurred during the face training, err: {ex}')
        result = False

    if result:
        internal_eb = EventBus(EventChannels.frtc)
        internal_eb.publish_async(json.dumps({'reloaded': result}))

    event_bus = EventBus(EventChannels.fr_train_response)
    event = json.dumps({'result': result})
    event_bus.publish(event)
    logger.info('training complete and the event was published')
