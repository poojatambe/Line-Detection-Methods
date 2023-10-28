import gradio as gr
from line_detection import LineDetectMethods


def main(image_path, method):
    methods = LineDetectMethods(image_path)
    line_method = {
        'Hough transform': methods.hough_line_detect(),
        'Probabilistic Hough transform': methods.hough_linep_detect(),
        'Convolution Mask': methods.masked_line_detect(),
        'pylsd': methods.lsd_detect(),
        'Opencv LSD': methods.opencv_lsd_detect(),
        'MLSD': methods.mlsd_detect()
    }
    out_img = line_method.get(method)
    print(out_img, method)
    return out_img


demo = gr.Interface(
        main,
        inputs=[gr.Image(label='Upload Image', type='filepath'),
                gr.Radio(['Hough transform', 'Probabilistic Hough transform',
                          'Convolution Mask', 'pylsd', 'Opencv LSD', 'MLSD'],
                          type='value', label='Methods',
                          info='Select line detection method')],
        outputs=[gr.Image(label='Output')],
        allow_flagging='never',
        title='Line Detection Methods Demo',
        examples=[['./sample_image.jpg']]

)


demo.launch()
