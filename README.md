# Decision Tree

This is a repository containing a Decision Tree classifier for documents. More information on decision tree can be found [here](https://en.wikipedia.org/wiki/Decision_tree). 

The classifier is built from ground up; many existing packages provide the same Decision Tree functionality (such as ```sklearn```). This repository is my own implementation of the model The model takes as input vectors of documents in the form of: 

[```feature_1_freq, feature_2_freq... feature_n_freq```]

 Each element in the vector corresponds to a feature (token) in the vocabulary, and the value of each element is the frequency of that feature. A vectorizer for any given document will be provided in future updates.


This implementation uses [information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) as the feature selection criteria. 

## Usage

The main script that runs the Decision Tree classifier is ```run.py```:

```python
from model import DecisionTree
import build_dt

# define paths to training and testing data
TRAIN_PATH = "train.vectors.txt"
TEST_PATH = "test.vectors.txt"
```

Provide the paths to train and test sets before running ```run.py```.

## Setting Up Model
The model takes two arguments to control its learning. The variable ```MAX_DEPTH``` controls the number of levels of the decision tree, while ```MIN_GAIN``` controls the minimum amount of information gained from observing a feature.

In ```build_dt.py```:

```python
from dtnode import DTNode
from util import compute_info_gain
from model import DecisionTree

MAX_DEPTH = 50
MIN_GAIN = 0
```

## Output
The model will output a confusion matrix and the train and test accuracies.
```
Confusion matrix for the test data:
row is the truth, column is the system output

             talk.politics.guns talk.politics.mideast talk.politics.misc
talk.politics.guns 900 0 0 
talk.politics.mideast 8 891 1 
talk.politics.misc 39 43 818 

 Test accuracy=0.9662962962962963
```
