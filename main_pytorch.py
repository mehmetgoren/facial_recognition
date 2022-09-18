import logging
import time
import warnings

from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.read_service_event_handler import ReadServiceEventHandler
from core.train_event_handler import TrainEventHandler
from core.utilities import register_detect_service, start_thread, EventChannels


def main():
    warnings.filterwarnings('ignore')
    _ = register_detect_service('pytorch_facial_recognition_service', 'face_recognition_pytorch-instance', 'The PyTorch Facial Recognition Service®')

    def train_event_handler():
        logger.info('pytorch face training event handler will start soon')
        eb = EventBus(EventChannels.fr_train_request)
        th = TrainEventHandler()
        eb.subscribe_async(th)

    start_thread(fn=train_event_handler, args=[])

    handler = ReadServiceEventHandler()
    handler.start_listen_train_complete_event()

    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logger.info('pytorch face recognition service will start soon')

    def listen_event_bus():
        try:
            event_bus = EventBus(EventChannels.read_service)
            event_bus.subscribe_async(handler)
        except BaseException as ex:
            logger.error(f'an error occurred while listening read service handler, ex: {ex}')
            time.sleep(1.)
            listen_event_bus()

    listen_event_bus()


if __name__ == '__main__':
    main()
