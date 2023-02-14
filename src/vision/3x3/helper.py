import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, scores2, mean_scores2):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.subplot(121)
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.subplot(122)
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores2)
    plt.plot(mean_scores2)
    plt.ylim(ymin=0)
    plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    plt.text(len(mean_scores2)-1, mean_scores2[-1], str(mean_scores2[-1]))
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(.1)
