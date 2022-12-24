import matplotlib.pyplot as plt

def loss(train_counter, test_counter, train_losses, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig

def predictions(example_data, output):
    fig = plt.figure()
    for i in range(1):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[0][0], cmap='gray', interpolation='none')
        plt.title("asd: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    fig