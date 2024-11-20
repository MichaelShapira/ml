import numpy as np
import torch, os, json, base64, cv2, time
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    env = os.environ
    model = YOLO(os.path.join(model_dir, env['YOLOV11_MODEL']))
    return model

def input_fn(request_body, content_type='application/x-image'):
    if content_type == 'application/x-image':
        nparr = np.frombuffer(request_body, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image 
        
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
    
def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        print("Making prediction")
        result=model.predict(source=input_data,conf=0.20,line_width=1)    
    return result
        
def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    infer = {}
    for result in prediction_output:
        if 'boxes' in result._keys and result.boxes is not None:
            infer['boxes'] = result.boxes.cpu().numpy().data.tolist()
        if 'masks' in result._keys and result.masks is not None:
            infer['masks'] = result.masks.cpu().numpy().data.tolist()
        if 'keypoints' in result._keys and result.keypoints is not None:
            infer['keypoints'] = result.keypoints.cpu().numpy().data.tolist()
        if 'probs' in result._keys and result.probs is not None:
            infer['probs'] = result.probs.cpu().numpy().data.tolist()
    return json.dumps(infer)