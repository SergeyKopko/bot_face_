import logging
from logging.handlers import RotatingFileHandler

from bot_sheduler import *
from constants import LOG_DATETIME_FORMAT, LOG_DIR, LOG_FORMAT
from sheduler.QueueState import QueueState


queue_state = QueueState()

LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "main.log"
rotating_handler = RotatingFileHandler(
    log_file, maxBytes=10**8, backupCount=5
)
logging.basicConfig(
    datefmt=LOG_DATETIME_FORMAT,
    format=LOG_FORMAT,
    level=logging.INFO,
    handlers=(rotating_handler, logging.StreamHandler()),
)
logging.getLogger("apscheduler.executors.default").disabled = True
logging.getLogger("apscheduler.scheduler").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("py.warnings").setLevel(logging.ERROR)

if __name__ == "__main__":
    logging.info("Shedule starting...")

import asyncio

async def main():
    scheduler.start()
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

