# Capstone Project

## Objective and Guidelines
The capstone project is to write a transformer-based model that can write python code (with proper whitespace indentations). Here are the key points:

1. The dataset can be found [here](https://drive.google.com/file/d/1rHb0FQ5z5ZpaY2HpyFGY6CeyDG0kTLoO/view?usp=sharing). The dataset has 4600+ examples of English text to Python code.
2. Write a transformer-based model that can learn to translate English to Python.
3. Must use transformers with self-attention, multi-head, and scaled-dot product attention in the model
4. There is no limit on the number of training epochs or total number of parameters in the model
5. Train a separate embedding layer for python keywords that pays special attention to whitespaces, colon and other things (like comma etc)
6. Your model has failed the capstone score if:
    * your model fails to do proper indentation
    * your model fails to use newline properly
    * your model has failed to understand how to use colon (:)
    * your model has failed to generate proper python code that can run on a Python interpreter and produce proper results
7. You need to take care of some preprocessing things like:
    * the dataset provided is divided into English and "python-code" pairs properly
    * the dataset does not have anomalies w.r.t. indentations (like a mixed-use of tabs and spaces, or use of either 4 or 3 spaces, it should be 4 spaces only). Either use tabs only or 4 spaces only, not both
    * the length of the "python-code" generated is not out of your model's capacity
8. You need to submit a detailed README file that:
    * describes the data clean required
    * your model architecture and salient features w.r.t. the model we wrote in the class
    * describes the loss function and how is it unique or improved than just using cross-entropy
    * your data preparation strategy
    * your data extension strategy (if you add more data)
    * your "python-code" embedding strategy
    * your evaluation metrics
    * 25 example output from your model
    * attention graph/images between text and "python-code"
    * any additional point you'd want to add

## Capstone Solution
Our objective is to generate python source code given a natural language query in English Language.

## About the Dataset

The dataset contains natural language queries and python code snippets pair manually annotated. The raw data contains around 4600 samples. The raw data can be downloaded from [here](https://github.com/anubhabPanda/END_Phase1/blob/main/Week14/Dataset/english_python_data.txt).

The raw data had the following issues:
  * In some cases the indentation was absent
  * There wasn't any space separating two samples

Few examples of raw data :

**Example 1**

    # write a python program to add two numbers 
    num1 = 1.5
    num2 = 6.3
    sum = num1 + num2
    print(f'Sum: {sum}')

**Example 2**

    # write a python program to add two numbers 
    num1 = 1.5
    num2 = 6.3
    sum = num1 + num2
    print(f'Sum: {sum}')

The raw data was manually cleaned to fix this issue. The cleaned dataset can be found [here](https://github.com/anubhabPanda/END_Phase1/blob/main/Week14/Dataset/english_python_data_corrected.txt)

## Data Preprocessing Steps
This step will convert the training examples in text file into a dataframe containing the natural language and source code in two columns. Steps taken :
  * The text file is read line by line and each line is stored in a list.
  * Iterating through the list we store the text between \n followed by #. 
  * The above steps will fail if comments are present in the text. Hence we take care of the comments by either removing them using regex or taking care in the data cleaning step.
  * Comments that are not preceded by a \n is retained
  * After converting to a dataframe, the natural language prompt are preceded by # and numbers. We remove # and numbers preceding the natural language queries. Space from the start and end of the text is also stripped.
  * We also remove data points where the length of the target source code text is greater than 250. We do this due to limited computational capacity. Also 80% of source code text is within 250 characters in length.
  * Number of data samples after cleaning and preprocessing = 3535

Data preprocessing is shown in the image below :

![Data Preprocessing](Images/data_preprocessing.png)

The final dataframe that will be used for modeling is shown below :

![Processed Data](Images/data_dataframe.png)

## Modeling Steps

We will tackle this problem step by step with incremental improvements in each step.

**Train-Validation-Test split used : 80-15-5** 

### **Baseline**
The code for this section can be found [**here**](https://github.com/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Baseline_Model.ipynb)

Alternate link for the above notebook can be found [**here**](https://nbviewer.jupyter.org/github/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Baseline_Model.ipynb)

In the first experiment we will setup a baseline. In the subsequent steps we will to this baseline model to improve the performance. For baseline we will be using the following:

* Tokenization : We will use default spacy english language model.
* Hyperparameters: 
  * Batch Size: 64
  * Encoder Layers: 3
  * Decoder Layer: 3
  * Hidden Dim: 256
  * Encoder PF Dimension: 512
  * Decoder PF Dimension: 512
  * Loss Function : Cross Entropy
  * Optimizer : Adam
  * Learning Rate: 0.0001
* Total Number of Parameters : 7.7 million
* Result of the baseline model: 
  * Training loss: 0.423
  * Validation loss: 2.229
  * Test loss: 2.416
* Few Outputs from the model:
  ![](Images/Exp_2_Result.png)
* Some problems with default spacy tokenizer :
    1. Opening bracket "(" in function is not tokenized as a separate token. Refer image below.
    2. "\n" and indentation(4 spaces) are tokenized as a single token
    3. Spacy default tokenization doesn't treat space(\s) as a separate token.

![default tokenizer](Images/default_tokenizer.png)


### **Experiment1**
The code for this section can be found [**here**](https://github.com/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment1.ipynb)

Alternate link for the above notebook can be found [**here**](https://nbviewer.jupyter.org/github/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment1.ipynb)

In the first experiment we will improve upon the baseline model. In this experiment we will use a custom tokenizer using spacy to take care of the tokenization problems mentioned in the earlier section. We will also increase the capacity of the model by increasing the number of parameters especially the decoder part. The features of this experiment and its result are shown below:

* Tokenization : We will use custom spacy tokenizer. The code and few examples from the custom tokenizer can be found below.
![custom tokenizer](Images/custom_tokenizer.png)
![token result](Images/custom_tokenizer_example.png)
![token result](Images/custom_tokenizer_example2.png)
* Hyperparameters: 
  * Batch Size: 64
  * Encoder Layers: 3
  * Decoder Layer: 6
  * Hidden Dim: 512
  * Encoder PF Dimension: 1024
  * Decoder PF Dimension: 1024
  * Loss Function : Cross Entropy
  * Optimizer : AdamW
  * Learning Rate: 0.0001
* Total Number of Parameters : 31 million
* Result of the model: 
  * Training loss: 0.398
  * Validation loss: 1.344
  * Test loss: 1.439
  ![Losses](Images/Exp_1_losses.png)
* Few Outputs from the model:
  ![exp result](Images/Exp_1_Result.png)
* Some shortcomings of the model and next steps:
    1. The model seems to be overfitting.
    2. Also we can try a lr scheduler to speed up the training.
    3. We can try out other loss functions to take care of overfitting.
    4. We can try to train the embedding layer separately to make the model converge faster.
    5. Still some room for improvement is there for the tokenizer.
    6. Add more data

### **Experiment2**
The code for this section can be found [**here**](https://github.com/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment2.ipynb)

Alternate link for the above notebook can be found [**here**](https://nbviewer.jupyter.org/github/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment2.ipynb)

In this experiment we will improve upon the previous model. In this experiment we will use a custom tokenizer using pygments library. We will try one cycle policy to train our model faster in each epoch. We will also try to reduce overfitting by increasing dropout a bit. The features of this experiment and its result are shown below:

* Tokenization : We will use custom tokenizer using pygments library. The code and few examples from the custom tokenizer can be found below.
  
  ![custom tokenizer](Images/pygments_tokenizer.png)
  ![token result](Images/pygments_tokenizer_example1.png)
  ![token result](Images/pygments_tokenizer_example2.png)

* Hyperparameters: 
  * Batch Size: 64
  * Encoder Layers: 4
  * Decoder Layer: 4
  * Hidden Dim: 512
  * Encoder PF Dimension: 1024
  * Decoder PF Dimension: 1024
  * Loss Function : Cross Entropy
  * Optimizer : Adam
  * Learning Rate: 0.00001 to 0.0001 (max lr at 30 epochs)
  * Epochs: 100
  * Dropout: 0.2
* Total Number of Parameters : 25.6 million
* Result of the model: 
  * Training loss: 0.515
  * Validation loss: 1.248
  * Test loss: 1.359
  
  ![Losses](Images/Exp_2_losses.png)
* Few Outputs from the model:
  ![exp result](Images/Exp_2_Result.png)
* Some observations from the model and next steps:
    1. We have reduced overfitting significantly
    2. One Cycle Learning rate seems to be giving stable results
    3. We can try different loss functions and metrics in further experiments

### **Experiment3**
The code for this section can be found [**here**](https://github.com/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment3.ipynb)

Alternate link for the above notebook can be found [**here**](https://nbviewer.jupyter.org/github/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment3.ipynb)

In this experiment we will try custom loss function and add more data. In this experiment we will use same tokenizer as previous experiment. We will try one cycle policy to train our model faster in each epoch. The features of this experiment and its result are shown below:

* Tokenization : We will use custom tokenizer using pygments library as in the previous experiment.
  
  ![custom tokenizer](Images/pygments_tokenizer.png)
  ![token result](Images/pygments_tokenizer_example1.png)
  ![token result](Images/pygments_tokenizer_example2.png)
* More Data: Added Data from CoNaLa dataset (code natural language)
* Hyperparameters: 
  * Batch Size: 64
  * Encoder Layers: 3
  * Decoder Layer: 3
  * Hidden Dim: 512
  * Encoder PF Dimension: 1024
  * Decoder PF Dimension: 1024
  * Loss Function : Label Smoothing Cross Entropy
    ![Label Smoothing Loss function](Images/Label_Smoothing_Loss.png)
  * Optimizer : Adam
  * Learning Rate: 0.00001 to 0.0001 (max lr at 30 epochs)
  * Epochs: 100
  * Dropout: 0.2
* Total Number of Parameters : 22.9 million
* Result of the model: 
  * Training loss: 1.538
  * Validation loss: 2.216
  * Test loss: 2.238
  
  ![Losses](Images/Exp_3_losses.png)

* Some observations from the model and next steps:
    1. Label Smoothing loss function doesn't reduce the loss any further than in experiment 2
    2. Even adding more data from CoNaLa corpus doesn't improve the loss although the comparison is not apple to apple as the number of samples in train, validation and test set have increased.

### **Conclusion and Final Results**

* From the baseline and 3 experiments, it becomes clear that Experiment 2 notebook gave the best results.
* Label Smoothing Cross Entropy Loss function although reduces overfitting , still the loss is high as compared to Cross Entropy
* I didn't train the embeddings layer separately as I didn't think it would help in improving performance. It will only reduce the training time and help the model converge faster as the initial epochs of the model are used to learn the Embeddings layer. I have used Pytorch embedding and it is trained in the task itself.
* Adding more data from the CoNaLa corpus didn't imrpove  the result. This may have happened since the both datasets are a bit different. CoNaLa dataset contains only single line python code whereas our dataset contains multiple lines as well.
* Custom Tokenizer and One Cycle LR helped to improve the loss a lot as seen in Experiment 1 and Experiment 2.
* Please refer to [**this**](https://nbviewer.jupyter.org/github/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment2.ipynb) notebook for the final results using all the learnings from the 4 experiments
  
25 outputs from the model can be found in [**same**](https://nbviewer.jupyter.org/github/anubhabPanda/END_Phase1/blob/main/Week14/Notebooks/Experiment2.ipynb) notebook

### Further Improvements

* Try Beam Search for decoding during validation, test and inference
* Try other Transformer Architectures like T5
* Add more data by making multiple dataset similar to our objective