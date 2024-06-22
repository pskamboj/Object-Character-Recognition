import cv2
import cvlib as cv
import urllib.request
import typing
import numpy as np
from paddleocr import PaddleOCR
import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
import urllib.parse
import urllib.request
import json

url = 'http://192.168.29.205/cam-hi.jpg'
esp32_url = 'http://192.168.29.11/recognized_text' 


enhancer = PaddleOCR(use_angle_cls=True, lang='en')


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":
    
    configs = BaseModelConfigs.load(r"Models\03_handwriting_recognition\202403291029\configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    accum_cer = []
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    image_path = cv2.imdecode(imgnp, -1)

    label = ""  # Assuming label is empty for now
    image = image_path
    resized_img = cv2.resize(image, (228, 132))
    prediction_text = model.predict(image)

    cer = get_cer(prediction_text, label)
    result = enhancer.ocr(resized_img, cls=True)
    recognized_text = []
    for line in result:
        for word in line:
            recognized_text.append(word[1][0])
    recognized_text_str = ' '.join(recognized_text)

    print(f"Image:{image_path}, Label:{label}, Prediciton:{recognized_text}, CER:{cer}")
    # accum_cer.append(cer)

    # Send recognized_text to ESP32-CAM module using urllib
    data = {"recognized_text": recognized_text_str}
    data = urllib.parse.urlencode(data)
    data = data.encode('utf-8')  # Convert data to JSON format and encode it
    req = urllib.request.Request(esp32_url, data=data,method='POST')
    print("HTTP Method:", req.method)
    try:
        with urllib.request.urlopen(req) as response:
            print("Text sent successfully to ESP32-CAM module")
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.reason)
    except urllib.error.URLError as e:
        print("URL Error:", e.reason)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Average CER:{np.average(accum_cer)}")
