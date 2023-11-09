import argparse
import json

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

import src.tools
from src.build_models import build_regression, build_classification


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! DRAW parser', add_help=True)

    parser.add_argument('--yolo_path', default='./trained_models/yolo_ygo.pth', type=str,
                        help="Path to trained yolo model")
    parser.add_argument('--source', default='', type=str,
                        help="Path to source video")
    parser.add_argument('--deck_list', default='', type=str,
                        help="Path to deck list file")
    parser.add_argument('--data_path', default='./cardDatabaseFormatted', type=str,
                        help="Path to formatted data base")

    return parser.parse_args()


def main(args):
    card_types = ['spell', 'monster', 'link', 'xyz', 'syncro', 'trap']

    with open("config.json", "rb") as f:
        configs = json.load(f)
    with open(args.deck_list) as f:
        deck_list = [line.rstrip() for line in f.readlines()]

    model_regression = build_regression(args.yolo_path)
    model_classification_dict, classes_dict, deck_card_ids = build_classification(
        card_types=card_types,
        configs=configs,
        data_path=args.data_path,
        deck_list=deck_list
    )

    results = model_regression(
        source=args.source,
        show_labels=False,
        save=False
    )

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter('filename.avi', fourcc, 30, (1280, 720))
    try:
        for frame, result in enumerate(results):
            image = result.orig_img.copy()
            for nbox, boxes in enumerate(result):
                x1, y1, x2, y2, = map(int, boxes.boxes.xyxy.squeeze())
                roi = image[y1:y2, x1:x2]
                contours = src.tools.extract_contours(roi.copy())

                min_area = 2000
                max_area = 4000

                if contours != ():
                    contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                    area = cv2.contourArea(contour)

                    if min_area < area < max_area:
                        box_artwork, box_txt = src.tools.extract_artwork(contour, x2 - x1, y2 - y1)
                        if box_artwork is None:
                            break

                        card_type = src.tools.get_card_type(roi.copy(), card_types, box_artwork, box_txt, configs)

                        if cv2.contourArea(box_artwork) > 10000:
                            dx = box_artwork[3][1] - box_artwork[0][1]
                            dy = box_artwork[3][1] - box_artwork[0][1]
                            artwork = src.tools.crop_min_area_rect(roi.copy(), box_artwork,
                                                                   180 - np.arctan(dx / dy) * 180 / np.pi)

                            if artwork.shape[0] != 0 and artwork.shape[1] != 0:
                                cv2.imwrite('./ROI/artwork.png', artwork)
                                input_image = src.tools.pil_loader('./ROI/artwork.png')
                                input_image = transforms.ToTensor()(input_image)
                                input_image = transforms.Resize((224, 224), antialias=True)(input_image)
                                artwork = transforms.ToPILImage()(input_image)
                                model_classification = model_classification_dict[card_type]
                                model_classification.eval()
                                input_tensor = input_image.unsqueeze(0).to('cuda')
                                output = model_classification(input_tensor)

                                _, indices = torch.sort(output, descending=True)
                                for k, i in enumerate(indices[0]):
                                    if i in deck_card_ids[card_type]:
                                        artwork.save('./ROI/frame_{}_box_{}_{}.png'.format(
                                            frame, nbox, classes_dict[card_type][i]
                                        ))
                                        cv2.putText(image, classes_dict[card_type][i], (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                                                    1.0,
                                                    (255, 255, 255), 2)

                                        print()
                                        print(classes_dict[card_type][i])
                                        break
                                cv2.drawContours(roi, [box_artwork], 0, (119, 152, 255), 2)
            video_writer.write(image)

    except KeyboardInterrupt:
        video_writer.release()


if __name__ == '__main__':
    main(parse_command_line())
