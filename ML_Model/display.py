import matplotlib.pyplot as plt

def predictions(example_data, output):
    index = 5
    input = example_data[index][0]

    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.tight_layout()
    plt.imshow(input, cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][index].item()))
    plt.xticks([])
    plt.yticks([])
    fig