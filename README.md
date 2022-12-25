
# Online MNIST

An MNIST model deployed online as a full stack application.  
Potential Roadmap:  
    - Deploy as service using Fly.io or github pages  
    - Further train my model with more epochs and datasets  


## Tech Stack

**Front End:** Javascript/HTML/CSS

**Backend:** Python: Flask, Pytorch

**Libraries/Dependencies:**
    Ajax, Axios, 
    Matplotlib, PIL


## Demo

Insert gif or link to demo

## Model Details

The model is small and simple.  
In brief the model used:  
    -A Convolutional Neural Network (CNN)  
    -3 epochs with a learning rate of 0.01  
    -Train batch size of 64, Test batch size of 1000  

On the MNIST dataset, the model performed quite well, with a 97% accuracy:  

![App Screenshot](https://user-images.githubusercontent.com/119146767/209481036-9490413a-34d7-43b5-a87e-4f6f24a6a8ec.png)
  
However, the model struggled on the data sent through the web app. Potential issues could be: lack of training, issues with converting HTML5 Canvas images to testing images, or more.

## Usage

Clone the repo

```bash
    git clone https://github.com/d0ngeun/Online-MNIST.git
    cd Online-MNIST
```

Train the model
```bash
    cd ML_Model
    python3 model.py
```

Launch the Flask app
```bash
    cd..
    python -m flask --app server run
```    

## Acknowledgements

 - https://nextjournal.com/gkoehler/pytorch-mnist
 - https://papers.nips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf