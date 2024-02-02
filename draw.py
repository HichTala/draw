import json
import os

import cv2
import numpy as np
import torch

import src.tools
from src.build_models import build_regression, build_classification


class Draw:
    def __init__(self, config, deck_list, source):
        with open(config, "rb") as f:
            self.configs = json.load(f)
        with open(deck_list) as f:
            self.deck_list = [line.rstrip() for line in f.readlines()]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.card_types = self.configs['card_types']
        model_regression = build_regression(os.path.join(self.configs['trained_models'], 'yolo_ygo.pt'))
        self.model_classification_dict, self.classes_dict, self.deck_card_ids = build_classification(
            card_types=self.card_types,
            configs=self.configs,
            data_path=self.configs['data_path'],
            deck_list=self.deck_list,
            device=device
        )

        self.results = model_regression(
            source=source,
            show_labels=False,
            save=False,
            device=device,
            stream=True,
            verbose=False
        )

    def process(self, result, display=False):
        predictions = []

        image = result.orig_img.copy()
        for nbox, boxes in enumerate(result):
            x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())

            box_min_area = self.configs['box_min_area']
            box_max_area = self.configs['box_max_area']

            if box_min_area < np.abs((x1 - x2) * (y1 - y2)) < box_max_area:
                roi = image[y1:y2, x1:x2]

                contours = src.tools.extract_contours(roi.copy())

                min_area = self.configs['txt_min_area']
                max_area = self.configs['txt_max_area']

                if contours != ():
                    contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                    area = cv2.contourArea(contour)

                    if min_area < area < max_area:
                        box_artwork, box_txt = src.tools.extract_artwork(contour, x2 - x1, y2 - y1)
                        if box_artwork is None:
                            break

                        if cv2.contourArea(box_artwork) > self.configs['area_threshold']:

                            angle = src.tools.get_angle(box_artwork)
                            artwork = src.tools.crop_min_area_rect(
                                roi.copy(),
                                box_artwork,
                                angle
                            )

                            if artwork.shape[0] != 0 and artwork.shape[1] != 0:
                                cv2.imwrite('./ROI/artwork.png', artwork)
                                card_type = src.tools.get_card_type(
                                    roi=roi,
                                    card_types=self.card_types,
                                    box_artwork=box_artwork,
                                    box_txt=box_txt,
                                    configs=self.configs
                                )

                                input_image = src.tools.pil_loader('./ROI/artwork.png')
                                transform = src.tools.build_transform()
                                input_image = transform(input_image)

                                model_classification = self.model_classification_dict[card_type]
                                model_classification.eval()

                                input_tensor = input_image.unsqueeze(0).to('cuda')
                                output = model_classification(input_tensor)

                                _, indices = torch.sort(output, descending=True)
                                for k, i in enumerate(indices[0]):
                                    if i in self.deck_card_ids[card_type]:
                                        predictions.append((self.classes_dict[card_type][i], card_type))
                                        cv2.putText(image, self.classes_dict[card_type][i], (x1, y1),
                                                    cv2.FONT_HERSHEY_PLAIN,
                                                    1.0,
                                                    (255, 255, 255),
                                                    2)
                                        break
                                cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 152, 119), thickness=2)
                                cv2.drawContours(roi, [box_txt], 0, (152, 255, 119), 2)
                                cv2.drawContours(roi, [box_artwork], 0, (119, 152, 255), 2)
        if display:
            return image
        else:
            return predictions
