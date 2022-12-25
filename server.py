import os
import sys
import json
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageFilter
from io import BytesIO
import base64
import json
import re

import torch
import torchvision
from torchvision import transforms
from ML_Model.net import CNN
from matplotlib import pyplot as plt
import cv2
import numpy as np


app = Flask(__name__)

def imageprepare(im):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva

@app.route('/', methods=['GET', 'POST'])
def paintapp():
    if request.method == 'GET':
        return render_template("paint.html")    

@app.route('/hook', methods=['POST'])
def save_canvas():
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.save('images/class1/canvas.png')
    print('Image received: {}'.format(im.size), file=sys.stderr)


    model = CNN()
    model.load_state_dict(torch.load("results/model.pth"))
    model.eval()

    test_dataset = torchvision.datasets.ImageFolder('images/', transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                            transforms.Resize((28, 28)),
                                                                            transforms.ToTensor(),
                                                                            transforms.Normalize((0.1307,), (0.3081,))]))

    examples = enumerate(test_dataset)
    batch_idx, (example_data, example_targets) = next(examples)
    predictions = model(example_data)

    pred_num = "Prediction: {}".format(predictions.data.max(1, keepdim=True)[1][0].item())
    print(pred_num)

    return pred_num

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template("search.html")
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)