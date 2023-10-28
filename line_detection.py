import cv2
import numpy as np
from pylsd import lsd
import tensorflow as tf
import argparse


class LineDetectMethods:
    def __init__(self, image_path):
        self.image_path = image_path

    def hough_line_detect(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, 1, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 200)
        lines = cv2.HoughLines(canny, rho=1, theta=np.pi/180, threshold=100)
        # print(lines)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite('hough_line.jpg', img)
        return img

    def hough_linep_detect(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, 1, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 200)
        lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi/180, threshold=100)
        print(lines)
        for line in lines:
            [x1, y1, x2, y2] = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite('hough_linep.jpg', img)
        return img

    def masked_line_detect(self):
        img = cv2.imread(self.image_path)
        horizontal_mask = np.array([[-1, -1, -1],
                              [2, 2, 2],
                              [-1, -1, -1]])
        vertical_mask = np.array([[-1, 2, -1],
                              [-1, 2, -1],
                              [-1, 2, -1]])
        oblique_plus_45_mask = np.array([[-1, -1, 2],
                              [-1, 2, -1],
                              [2, -1, -1]])
        oblique_minus_45_mask = np.array([[2, -1, -1],
                              [-1, 2, -1],
                              [-1, -1, 2]])

        # Blur the image to reduce noise.
        blurred_image = cv2.GaussianBlur(img, (5, 5), 0)

        # Convolution with horizontal mask.
        horizontal_filtered_image = cv2.filter2D(blurred_image,
                                                 -1, horizontal_mask)

        #   Convolution with vertical mask.
        vertical_filtered_image = cv2.filter2D(blurred_image,
                                               -1, vertical_mask)

        # Convolution with oblique (+45 degrees) mask.
        oblique_plus_45_filtered_image = cv2.filter2D(blurred_image,
                                                      -1, oblique_plus_45_mask)

        # Convolution with oblique (-45 degrees) mask.
        oblique_minus_45_filtered_image = cv2.filter2D(blurred_image,
                                                    -1, oblique_minus_45_mask)

        cv2.imwrite('horizontal_lines.jpg', horizontal_filtered_image)
        cv2.imwrite('slant_lines.jpg', oblique_plus_45_filtered_image)
        cv2.imwrite('slant_lines1.jpg', oblique_minus_45_filtered_image)
        cv2.imwrite('vertical_lines.jpg', vertical_filtered_image)
        out = np.hstack([horizontal_filtered_image, vertical_filtered_image,
                        oblique_plus_45_filtered_image,
                        oblique_minus_45_filtered_image])
        return out

    def lsd_detect(self):
        img = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segments = lsd(img_gray, scale=0.5)
        for i in range(segments.shape[0]):
            pt1 = (int(segments[i, 0]), int(segments[i, 1]))
            pt2 = (int(segments[i, 2]), int(segments[i, 3]))
            width = segments[i, 4]
            cv2.line(img, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
        cv2.imwrite('pylsd_output.jpg', img)
        return img

    def opencv_lsd_detect(self):
        img = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(img_gray)[0]
        drawn_img = lsd.drawSegments(img, lines)
        cv2.imwrite('opencv_lsd_output.jpg', drawn_img)
        # cv2.imshow("LSD", drawn_img )
        # cv2.waitKey(0)
        return drawn_img

    def mlsd_detect(self):
        from mlsd.utils import pred_lines

        model_name = 'mlsd/tflite_models/M-LSD_512_large_fp32.tflite'
        interpreter = tf.lite.Interpreter(model_path=model_name)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = cv2.imread(self.image_path)
        score_thr = 0.2
        dist_thr = 50.0
        lines = pred_lines(img, interpreter, input_details,
                           output_details, input_shape=[512, 512],
                           score_thr=score_thr, dist_thr=dist_thr)

        for line in lines:
            x_start, y_start, x_end, y_end = [int(val) for val in line]
            cv2.line(img, (x_start, y_start), (x_end, y_end), [0, 0, 255], 2)
        cv2.imwrite('mlsd_output.jpg', img)
        # cv2.imshow('out', img)
        # cv2.waitKey(0)
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Specify input image',
                        default='./sample_image.jpg', type=str)
    parser.add_argument('--method', help='Specify line detection method name',
                        default='Hough transform', type=str)
    opt = parser.parse_args()
    detect = LineDetectMethods(opt.image)
    if opt.method == 'Hough transform':
        out_img = detect.hough_line_detect()
    elif opt.method == 'Probabilistic Hough transform':
        out_img = detect.hough_linep_detect()
    elif opt.method == 'Convolution Mask':
        out_img = detect.masked_line_detect()
    elif opt.method == 'pylsd':
        out_img = detect.lsd_detect()
    elif opt.method == 'OpenCV LSD':
        out_img = detect.opencv_lsd_detect()
    elif opt.method == 'MLSD':
        out_img = detect.mlsd_detect()
    else:
        print('Specify correct method.')
