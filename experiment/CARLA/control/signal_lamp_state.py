"""Current-image signal classification; static geometry contains no light states."""

import math
import numpy as np
import torch
from torch import nn

STATES = ('UNKNOWN', 'RED', 'YELLOW', 'GREEN')


def project_head(vertices, inverse_camera, width, height, fov):
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (8, 3) or not np.isfinite(points).all():
        return None
    camera = np.c_[points, np.ones(8)] @ np.asarray(inverse_camera).T
    if not np.isfinite(camera).all() or (camera[:, 0] <= .5).any():
        return None
    focal = width / (2. * math.tan(math.radians(fov) / 2.))
    x = width / 2. + focal * camera[:, 1] / camera[:, 0]
    y = height / 2. - focal * camera[:, 2] / camera[:, 0]
    box = [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
    if box[0] < 0 or box[1] < 0 or box[2] >= width or box[3] >= height:
        return None
    if box[2] - box[0] < 2 or box[3] - box[1] < 4:
        return None
    return box


def head_crop(rgb, box):
    import cv2
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    px, py = max(2., .25 * (x2-x1)), max(2., .15 * (y2-y1))
    crop = rgb[max(0, int(y1-py)):min(h, int(math.ceil(y2+py))),
               max(0, int(x1-px)):min(w, int(math.ceil(x2+px)))]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (48, 64), interpolation=cv2.INTER_LINEAR)


def color_evidence(rgb):
    import cv2
    # Use the lamp core, not the surrounding foliage or colored sign background.
    height,width=rgb.shape[:2]
    rgb=rgb[int(.1*height):max(int(.9*height),int(.1*height)+1),int(.25*width):max(int(.75*width),int(.25*width)+1)]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (s > 30) & (v > 100)
    return dict(RED=int((lit & ((h < 18) | (h > 168))).sum()),
                YELLOW=int((lit & (h >= 18) & (h <= 40)).sum()),
                GREEN=int((lit & (h >= 42) & (h <= 100)).sum()))


class LampStateNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in ((3, 16), (16, 32), (32, 48)):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 3)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(48*4*3, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, rgb):
        return self.head(self.features(rgb))


class LampStateClassifier:
    def __init__(self, checkpoint, device='cuda:0'):
        self.device = device
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['schema_version'] != 'lamp_state/1.0' or tuple(saved['classes']) != STATES:
            raise ValueError('Incompatible lamp-state checkpoint')
        self.model = LampStateNet().to(device).eval()
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.threshold = float(saved.get('confidence_threshold', .9))
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, 64, 48, device=device))

    def predict(self, crops):
        if not crops:
            return []
        batch = torch.from_numpy(np.stack(crops).transpose(0, 3, 1, 2).copy()).float().to(self.device) / 255.
        with torch.inference_mode():
            probs = self.model(batch).softmax(-1).cpu().numpy()
        output = []
        for crop, p in zip(crops, probs):
            index = int(p.argmax()); state = STATES[index]; confidence = float(p[index])
            counts = color_evidence(crop)
            # A model prediction cannot create a permissive state without current RGB evidence.
            if state != 'UNKNOWN' and (confidence < self.threshold or sum(counts.values()) < 3
                    or (state=='GREEN' and (counts['GREEN']<3 or counts['GREEN']>.24*crop.shape[0]*crop.shape[1]))):
                state, confidence = 'UNKNOWN', 0.
            output.append(dict(state=state, color_confidence=confidence,
                probabilities={k:float(v) for k,v in zip(STATES,p)}, color_pixels=counts,
                source='current_rgb_static_head_lamp_net'))
        return output
