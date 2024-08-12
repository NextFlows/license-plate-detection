import os
import sys
import time
import cv2
import traceback
from tqdm import tqdm

from model.license_plate_detector import LicensePlateDetector

if __name__=='__main__':
    input_path = sys.argv[1]
    device = 'cuda'

    if os.path.isdir(input_path):
        img_paths = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    else:
        img_paths = [input_path]

    os.makedirs('results', exist_ok=True)
    with open('result.csv', 'w') as f:
        f.write(f'img_path,detection\n')
        f.close()

    # Detector
    detector = LicensePlateDetector(use_cuda=True)
    tic = time.time()
    for img_path in tqdm(sorted(img_paths)):
        # Read image

        input_image = cv2.imread(img_path)
        img = input_image.copy()

        # Run Detections
        try:
            output = detector.detect(
                            image=input_image,
                            conf_thres=0.2,
                            input_shape=(640, 640)
            )

        except Exception as e:
            print('ERROR in file: ', img_path)
            print(traceback.format_exc())
            continue

        with open('result.csv', 'a') as f:
            f.write(f'{img_path},{output}\n')
            f.close()

    toc = time.time()
    print(f'Infer time {toc-tic}')
