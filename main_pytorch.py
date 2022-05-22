import warnings

from common.event_bus.event_bus import EventBus
from common.utilities import logger
from core.event_handlers import ReadServiceEventHandler
from core.utilities import register_detect_service


def main():
    warnings.filterwarnings('ignore')
    _ = register_detect_service('pytorch_detection_service', 'The PyTorch Facial Recognition Service®')

    handler = ReadServiceEventHandler()

    logger.info('pytorch object detection service will start soon')
    event_bus = EventBus('read_service')
    event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
