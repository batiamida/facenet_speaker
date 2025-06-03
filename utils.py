import asyncio
import base64
import queue
import threading

import cv2
import numpy as np
import torch
import websockets
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import os


def show_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def quantize_pytorch_model(model, input_data, model_name="resnet50_vggface2") -> str:
    """
    :param model: pytorch model to quantize
    :param input_data: example of input that .onnx model expect
    :param model_name: the name of the model for saving
    :param device: cpu or cuda depends on which device you want to use for quantization
    :return: path to quantized model
    """
    base_model = f"{model_name}.onnx"

    torch.onnx.export(
        model,
        input_data,
        base_model,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )


    onnx_model = onnx.load(base_model)
    onnx.checker.check_model(onnx_model)

    quant_model = f"{model_name}_quantized8.onnx"
    quantize_dynamic(base_model, quant_model, weight_type=QuantType.QUInt8)

    return os.path.join(os.getcwd(), quant_model)


class FrameReciever:
    def __init__(self, uri, token):
        self.uri = uri
        self.token = token
        self.q = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop)
        self.ws = None
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.create_task(self._receiver())
        self.loop.run_forever()

    async def _receiver(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        async with websockets.connect(self.uri, additional_headers=headers) as ws:
            self.ws = ws

            while not self.stop_event.is_set():
                try:
                    data = await ws.recv()
                    img_bytes = base64.b64decode(data)
                    nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.q.put(frame)
                except asyncio.TimeoutError:
                    continue

    def read(self):
        return True, self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        return self.q.get()

    def release(self):
        self.stop_event.set()

        # Close websocket
        if self.ws:
            close_future = asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            try:
                close_future.result(timeout=5)
            except Exception as e:
                print("WebSocket close failed:", e)

        # _receiver exiting
        if hasattr(self, 'receiver_task'):
            done_future = asyncio.run_coroutine_threadsafe(self.receiver_task, self.loop)
            try:
                done_future.result(timeout=5)
            except Exception as e:
                print("Receiver did not complete cleanly:", e)

        # stop the loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


if __name__ == "__main__":
    stream = FrameReciever(os.getenv("camera_link"), os.getenv("camera_auth_token"))