import abc
import mss, cv2, base64
import numpy as np
from typing import AsyncIterator, List, Dict, Any
from PIL import Image

class abc_llm(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    async def text_chat(self,
                      messages: List[Dict[str, Any]],
                      system: str = None,
                      ) -> AsyncIterator[str]:
        '''
        寫的時候繼承這個回傳結構也記得符合 ：Ｄ 秋秋加油
        :param messages:
        :param system:
        :return:
        ollama run llama3.2
        '''

        raise NotImplementedError

    @abc.abstractmethod
    async def image_chat(self,
                         image: Image.Image | str | bytes,
                         system: str = None,
                         ) -> AsyncIterator[str]:

        raise NotImplementedError

    @staticmethod
    def screen_shot(self):
        if self.MSS is None:
            self.MSS = mss.mss()

        # Take a screenshot of the main screen
        frame_bytes = self.MSS.grab(self.MSS.monitors[0]) #PRIMARY_MONITOR

        frame_array = np.array(frame_bytes)
        # resize
        frame_resized = cv2.resize(frame_array, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        result, frame_encoded = cv2.imencode('.jpg', frame_resized, encode_param)
        # base64
        frame_base64 = base64.b64encode(frame_encoded).decode("utf-8")
        return frame_base64