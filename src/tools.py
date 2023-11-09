import cv2
import numpy as np
from PIL import Image


def get_edges(num_edge, coordinates):
    if num_edge in [1, 2]:
        set1 = coordinates[np.argsort(coordinates[:, 0])][:2]
    else:
        set1 = coordinates[np.argsort(coordinates[:, 0])][2:]
    if num_edge in [1, 3]:
        set2 = coordinates[np.argsort(coordinates[:, 1])][:2]
    else:
        set2 = coordinates[np.argsort(coordinates[:, 1])][2:]

    h, w = set1.shape
    dtypes = {'names': ['f{}'.format(i) for i in range(w)],
              'formats': w * [set1.dtype]}

    intersection = np.intersect1d(set1.view(dtypes), set2.view(dtypes))
    intersection = intersection.view(set1.dtype).reshape(-1, w)[0]

    return np.where((coordinates == intersection).all(axis=1))[0][0]


def get_lower_edges(coordinates, h, w):
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)

    try:

        edge1 = get_edges(1, coordinates)
        edge2 = get_edges(2, coordinates)
        edge3 = get_edges(3, coordinates)
        edge4 = get_edges(4, coordinates)

        if coordinates[edge1][0] - coordinates[edge3][0] < coordinates[edge1][1] - coordinates[edge2][1]:
            if edge1 < h:
                return [coordinates[edge1], coordinates[edge2]], [coordinates[edge3], coordinates[edge4]]
            else:
                return [coordinates[edge2], coordinates[edge1]], [coordinates[edge4], coordinates[edge3]]
        else:
            if edge1 < w:
                return [coordinates[edge1], coordinates[edge3]], [coordinates[edge2], coordinates[edge4]]
            else:
                return [coordinates[edge3], coordinates[edge1]], [coordinates[edge4], coordinates[edge2]]

    except:
        return None, None


def crop_min_area_rect(roi, box, angle):
    h, w = roi.shape[0], roi.shape[1]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rot = cv2.warpAffine(roi, rotation_matrix, (w, h))

    pts = np.int0(cv2.transform(np.array([box]), rotation_matrix))[0]
    pts[pts < 0] = 0

    img_crop = img_rot[pts[:, 1].min():pts[:, 1].max(), pts[:, 0].min():pts[:, 0].max()]

    return img_crop


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def clean_deck_list(deck_list, card_type, classes):
    deck_card_id = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for card_id in deck_list:
        if card_id[0] in '0123456789':
            for card_name in class_to_idx.keys():
                if card_id[0] == '0':
                    if card_id[1:] in card_name:
                        deck_card_id.append(class_to_idx[card_name])
                if card_id in card_name:
                    deck_card_id.append(class_to_idx[card_name])
    return list(set(deck_card_id))


def extract_contours(roi):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    equalized = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(equalized, 140, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    edged = cv2.erode(thresh, kernel, iterations=3)
    edged = cv2.dilate(edged, kernel, iterations=3)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours


def extract_artwork(contour, width, height):

    rect = cv2.minAreaRect(contour)
    box_txt = cv2.boxPoints(rect)
    box_txt = np.intp(box_txt)
    # cv2.drawContours(roi_original, [box], 0, (152, 255, 119), 2)
    edges1, edges2 = get_lower_edges(box_txt, height, width)
    if edges1 is None:
        return None

    dx1 = edges1[1][0] - edges1[0][0]
    dx2 = edges2[1][0] - edges2[0][0]
    dy1 = edges1[1][1] - edges1[0][1]
    dy2 = edges2[1][1] - edges2[0][1]

    dst1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
    dst2 = (dx2 ** 2 + dy2 ** 2) ** 0.5

    box_artwork = np.array([
        [edges1[1][0] + dx1 * 5 / dst1, edges1[1][1] + dy1 * 5 / dst1],
        [edges2[1][0] + dx2 * 5 / dst2, edges2[1][1] + dy2 * 5 / dst2],
        [edges2[1][0] + dx2 * 95 / dst2, edges2[1][1] + dy2 * 100 / dst2],
        [edges1[1][0] + dx1 * 95 / dst1, edges1[1][1] + dy1 * 100 / dst1],
    ], dtype=np.int64)

    return box_artwork, box_txt


def get_card_type(roi, card_types, box_artwork, box_txt, configs):
    roi = roi.reshape((roi.shape[0] * roi.shape[1], 3)).astype(float)
    cv2.drawContours(roi, [box_artwork], 0, (1000, 1000, 1000), -1)
    cv2.drawContours(roi, [box_txt], 0, (1000, 1000, 1000), -1)

    counts = [
        np.isclose(
            np.array([np.linalg.norm(diff) for diff in (roi - configs[card_type]["colors"])]),
            np.zeros(roi.shape[0]),
            atol=configs[card_type]["tol"]
        ).sum() for card_type in card_types
    ]

    index_max = np.argmax(counts)
    return card_types[index_max]



