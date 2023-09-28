import os
import asyncio
import torch

from dotenv import load_dotenv
from fogverse import Consumer, ConsumerStorage, Producer
from fogverse.logging.logging import CsvLogging

load_dotenv()


class MyClientStorage(Consumer, ConsumerStorage, CsvLogging):
    def __init__(self, keep_messages=False):
        self.consumer_topic = ['result']
        self.consumer_servers = os.environ['LOCAL_KAFKA']
        Consumer.__init__(self)
        CsvLogging.__init__(self)
        ConsumerStorage.__init__(self, keep_messages=keep_messages)


async def main():
    _Consumer = MyClientStorage
    consumer = _Consumer()
    tasks = [consumer.run()]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.close()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
