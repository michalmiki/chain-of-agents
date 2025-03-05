"""
Example script for using Chain of Agents for summarization.
"""
import sys
import os

# Add the parent directory to sys.path to import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chain_of_agents import ChainOfAgents


def main():
    # Sample long text for summarization
    long_text = """
    Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.

    Neural networks help us cluster and classify. You can think of them as a clustering and classification layer on top of the data you store and manage. They help to group unlabeled data according to similarities among the example inputs, and they classify data when they have a labeled dataset to train on. (Neural networks can also extract features that are fed to other algorithms for clustering and classification; so you can think of deep neural networks as components of larger machine-learning applications involving algorithms for reinforcement learning, classification and regression.)

    What kind of problems does deep learning solve, and more importantly, can it solve yours? To know that, we need to think about which problems deep learning solves best.

    Deep-learning networks are distinguished from the more commonplace single-hidden-layer neural networks by their depth; that is, the number of node layers through which data must pass in a multistep process of pattern recognition.

    Previous iterations of neural networks were shallow, composed of one input and one output layer, and at most one hidden layer in between. More than three layers (including input and output) qualifies as "deep" learning. So deep is a strictly defined, technical term that means more than one hidden layer.

    In deep-learning networks, each layer of nodes trains on a distinct set of features based on the previous layer's output. The further you advance into the neural net, the more complex the features your nodes can recognize, since they aggregate and recombine features from the previous layer.

    This is known as feature hierarchy, and it is a hierarchy of increasing complexity and abstraction. It makes deep-learning networks capable of handling very large, high-dimensional data sets with billions of parameters that pass through nonlinear functions.

    Above all, these networks can discover latent structures within unlabeled, unstructured data, which is the vast majority of data in the world. Another word for unstructured data is raw media; i.e., images, text, video and audio. Deep learning can be applied to any form of data – to what the eye sees or the ear hears in order to distinguish between, say, dogs and cats, or between water dripping and fire crackling.

    With unstructured data, there may not be obvious features like the kinds we choose when we're doing classical machine learning. So deep learning is especially helpful in feature detection within images, which involves high dimensionality (each pixel is a dimension). For image recognition, you'll often see Convolutional Neural Networks (CNNs) in which we can learn features and we use known objects for finding new objects.

    A key advantage of deep learning networks is that they often continue to improve as the size of your data increases. In machine learning, you'll often hear that algorithms can have a performance ceiling, but deep learning models often have much higher ceilings, which makes them even more powerful as big data grows even bigger.

    However, deep learning networks are not a silver bullet. The advantage of feature learning comes at the cost of requiring much more data than traditional machine-learning algorithms. While a linear-regression model can be trained on a few dozen examples, deep networks often require thousands or millions of examples to achieve good performance.

    Beyond that, deep learning models (and neural networks more generally) are known as black-box models because they're typically very difficult to interpret. If you need to know why your model makes the predictions it does, deep learning might not be ideal. If all that matters is predictive performance and ample training data is available, deep learning may be suitable.

    Deep learning also presents computational and memory challenges. Training a deep network can take a very long time and require a lot of computational resources. This is especially true for networks with many layers, which can have many, many parameters.
    """
    
    # Initialize Chain of Agents
    coa = ChainOfAgents(verbose=True)
    
    # Generate a summary
    result = coa.summarize(long_text)
    
    # Print the result
    print("\n==== Final Summary ====")
    print(result["answer"])
    print("\n==== Metadata ====")
    print(f"Number of chunks: {result['metadata']['num_chunks']}")
    print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()
