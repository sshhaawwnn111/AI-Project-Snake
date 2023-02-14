import matplotlib.pyplot as plt
import numpy
from IPython import display
import numpy as np
plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    x = np.arange(2)
    plt.bar(x, [scores, mean_scores])
    plt.xticks(x, ['model1', 'model2'])
    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)
