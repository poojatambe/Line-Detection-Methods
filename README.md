# Line_detection_methods
The line detection is primary task in Computer Vision. There are different approaches to detect line from an image. Here will cover some of those methods with basic implementation.

## Methods:
* Hough transform
* Probabilistic Hough transform
* Convolution Mask
* pylsd
* OpenCV LSD
* MLSD

## Command:
```
! python line_detection.py --image image_name --method method_name
```
For method_name, refer above methods.

```
! python line_detection.py
```
This command will run line detection on sample image using hough transform.

![Alt text](image.png)

Run the demo.
```
python demo.py
```

## Requirements:
```
!pip install -r requirements.txt
```
For MLSD method, first clone https://github.com/navervision/mlsd.git repository.

## Results:
* Convolution Mask
![Alt text](result/horizontal_lines.jpg) ![Alt text](result/slant_lines.jpg)
![Alt text](result/slant_lines1.jpg)   ![Alt text](result/vertical_lines.jpg)

* Hough Transform
![Alt text](result/hough_line.jpg)

* Probabilistic Hough Transform
![Alt text](result/hough_linep.jpg)

* pylsd
![Alt text](result/pylsd_output.jpg)

* OpenCV LSD
![Alt text](result/opencv_lsd_output.jpg)

* MLSD
![Alt text](result/mlsd_output.jpg)

## Reference:
1. https://github.com/navervision/mlsd
2. https://github.com/AndranikSargsyan/pylsd-nova
3. https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html


