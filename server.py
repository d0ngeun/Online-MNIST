import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import re

import torch
import torchvision
from torchvision import transforms
from ML_Model.net import CNN
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def paintapp():
    if request.method == 'GET':
        return render_template("paint.html")    

@app.route('/hook', methods=['POST'])
def save_canvas():
    image_data = re.sub('^data:image/.+;base64,', '', request.form['imageBase64'])
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im.save('images/class1/canvas.png')
    im.close()
    return ''

@app.route('/api/pred')
def api_pred():
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
    pred_num = predictions.data.max(1, keepdim=True)[1][0].item()
    data = {
        'pred': pred_num,
    }

    return jsonify(data)

    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)