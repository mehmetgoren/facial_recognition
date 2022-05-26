import warnings

from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.read_service_event_handler import ReadServiceEventHandler
from core.train_event_handler import TrainEventHandler
from core.utilities import register_detect_service, start_thread, EventChannels


def main():
    warnings.filterwarnings('ignore')
    _ = register_detect_service('pytorch_detection_service', 'The PyTorch Facial Recognition Service®')

    def train_event_handler():
        logger.info('pytorch face training event handler will start soon')
        eb = EventBus(EventChannels.fr_train_request)
        th = TrainEventHandler()
        eb.subscribe_async(th)

    start_thread(fn=train_event_handler, args=[])

    handler = ReadServiceEventHandler()
    handler.start_listen_train_complete_event()

    logger.info('pytorch face recognition service will start soon')
    event_bus = EventBus(EventChannels.read_service)
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
