import os
import asyncio
import cv2
import torch

import numpy as np

from dotenv import load_dotenv
from fogverse import Consumer, Producer, ConsumerStorage
from fogverse.logging.logging import CsvLogging
from fogverse.util import get_header, numpy_to_base64_url

load_dotenv()

ENCODING = os.getenv('ENCODING', 'jpg')
MODEL = os.environ['MODEL_PATH']
WEIGHT = os.environ['WEIGHT_PATH']


class MyStorage(Consumer, ConsumerStorage):
    def __init__(self, keep_messages=False):
        self.consumer_topic = ['input']
        self.consumer_servers = os.environ['LOCAL_KAFKA']
        Consumer.__init__(self)
        ConsumerStorage.__init__(self, keep_messages=keep_messages)


class MyInference(CsvLogging, Producer):
    def __init__(self, consumer):
        # using cloned repo
        self.model = torch.hub.load(
            MODEL, 'custom', WEIGHT, source='local', force_reload=True)
        self.consumer = consumer
        CsvLogging.__init__(self)
        Producer.__init__(self)

    async def _after_start(self):
        # to fix the first inference bottleneck
        dummy = (np.random.rand(480, 640, 3)*255).astype(np.uint8)
        print('warming up')
        self.model(dummy)
        print('ready')

    async def receive(self):
        return await self.consumer.get()

    def _process(self, data):
        results = self.model(data)
        results.save()
        return results.render()[0]

    async def process(self, data):
        return await self._loop.run_in_executor(None,
                                                self._process,
                                                data)

    def encode(self, img):
        base64_url = numpy_to_base64_url(img, ENCODING)
        return base64_url.encode()

    async def send(self, data):
        headers = list(self.message.headers)
        headers.append(('type', b'final'))
        await super().send(data, headers=headers)
# ======================================================================


class MyJetsonSc2(MyInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def send(self, data):
        headers = list(self.message.headers)
        cam_id = get_header(headers, 'cam')
        headers.append(('from', b'jetson'))
        self.message.headers = headers
        self.producer_topic = f'final_{cam_id}'
        await super().send(data)
# ======================================================================


class MyJetsonSc4(MyInference):
    def __init__(self, *args, **kwargs):
        self.producer_topic = 'result'
        self.producer_servers = os.environ['LOCAL_KAFKA']
        super().__init__(*args, **kwargs)

    def decode(self, data):
        data = super().decode(data)
        if len(data.shape) == 1:
            # in case the data is encoded with cv2.imencode
            # case for compressed image
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data


scenarios = {
    2: (MyStorage, MyJetsonSc2),
    4: (MyStorage, MyJetsonSc4),
}


async def main():
    scenario = int(os.getenv('SCENARIO', 4))
    _Consumer, _Producer = scenarios[scenario]

    consumer = _Consumer()
    producer = _Producer(consumer)
    tasks = [consumer.run(), producer.run()]
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
