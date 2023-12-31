import asyncio
import os
import cv2

import numpy as np

from dotenv import load_dotenv
from pathlib import Path
from fogverse import Producer
from fogverse.logging import CsvLogging
from fogverse.util import compress_encoding, get_cam_id, get_timestamp_str, numpy_to_bytes

load_dotenv()


SIZE = (640, 480)
DIR = Path(os.environ['IMAGE_PATH'])


class MyFrameProducer(CsvLogging, Producer):
    def __init__(self, compress, compress_name, loop=None):
        self.cam_id = get_cam_id()
        self.producer_topic = 'input'
        self.frame_idx = 1
        self.compress = compress
        self.compress_name = compress_name
        CsvLogging.__init__(self)
        Producer.__init__(self, loop=loop)

    def _after_start(self):
        with open('images.txt') as f:
            self.images_name = f.read().split(',')

# ===================================================================================

    def _receive(self):
        img_name = self.images_name.pop(0)
        with open(str(DIR / img_name), 'rb') as f:
            data = f.read()
        return data

    async def receive(self):
        return await self._loop.run_in_executor(None, self._receive)

# ===================================================================================

    def _process(self, data):
        encode_func = self.compress[0]
        if not encode_func:
            return super().encode(data)
        encoded = encode_func(data, *self.compress[1:])
        return encoded

    async def process(self, data):
        return await self._loop.run_in_executor(None,
                                                self._process,
                                                data)

# ===================================================================================

    def decode(self, data):
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, SIZE)
        return frame

    async def send(self, data):
        key = str(self.frame_idx).encode()
        headers = [
            ('cam', self.cam_id.encode()),
            ('frame', str(self.frame_idx).encode()),
            ('timestamp', get_timestamp_str().encode())]
        await super().send(data, key=key, headers=headers)
        self.frame_idx += 1

# ===================================================================================


class MyFrameProducerSc2_4(MyFrameProducer):
    def __init__(self, compress, compress_name, loop=None):
        self.producer_servers = os.environ['LOCAL_KAFKA']
        super().__init__(compress, compress_name, loop)


class MyFrameProducerSc3(MyFrameProducer):
    def __init__(self, compress, compress_name, loop=None):
        self.producer_servers = os.getenv('CLOUD_KAFKA')
        super().__init__(compress, compress_name, loop)

# ===================================================================================


def process_grayscale(data, *args):
    processed = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    return numpy_to_bytes(processed)


processes = {
    'jpeg 50': [compress_encoding, 'jpg', (cv2.IMWRITE_JPEG_QUALITY, 50)],
    'jpeg 75': [compress_encoding, 'jpg', (cv2.IMWRITE_JPEG_QUALITY, 75)],
    'jpeg 95': [compress_encoding, 'jpg', (cv2.IMWRITE_JPEG_QUALITY, 95)],
    'jpeg 2000': [compress_encoding, 'jp2'],
    'grayscale': [process_grayscale],
    'original': [None]
}

scenarios = {
    2: (MyFrameProducerSc2_4),
    3: (MyFrameProducerSc3),
    4: (MyFrameProducerSc2_4),
}


async def main():
    compress_name = os.getenv('COMPRESSION', 'original')
    compress = processes[compress_name]
    scenario = int(os.getenv('SCENARIO', 4))

    _Producer = scenarios[scenario]
    producer = _Producer(compress, compress_name)
    tasks = [producer.run()]
    try:
        await asyncio.gather(*tasks)
    except:
        for t in tasks:
            t.close()

if __name__ == '__main__':
    asyncio.run(main())
