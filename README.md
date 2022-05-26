# simple-Bert-Retrieval-augmented-classification
Augmenting inputs to BERT, with retrieved information, before predicting their labels

### Main Idea:
The goal of this classification is to label passages as fake/real via a Bert model.
What is done differently is that in this experiment (vs normal classification), included Information Retrieval (via BM25) step to retrieve relevant pieces of information pertaining to the passage that is to be classified.
Subsequently, the retrieved information (only one) is/are fused with the passage by prepending retrieved info to Bert input.
This form of fusion done is the simplest and is called 'Priming'.
<br />

### Overview of Solution:
![image](https://user-images.githubusercontent.com/54625060/170184988-c058e8ba-687c-40e4-9515-2f32c5881012.png)
<br />

### Results
For the sake of reproducibility, the Prediction dataset, training code and seed(123) are kept same across the experiments, except the lines required to alter the fusion mechanism 
#### 1) Baseline without fusion
![single query](https://user-images.githubusercontent.com/54625060/170484839-3ab29a87-b6a9-4e08-a1cd-c320aa3f5e15.jpg)

<b>training file</b>: /src/training/submit_single_query_training.ipynb

<b>testing file</b>: /src/testing/submit_single_query(p)_testing.ipynb

<br />
#### 2) With fusion
![paired_query](https://user-images.githubusercontent.com/54625060/170484855-6b620cac-cfba-4181-8871-16b30ff58711.jpg)

<b>training file</b>: /src/training/submit_paired_query(p%2Ch)_training.ipynb

<b>testing file</b>: /src/testing/submit_paired_query(p%2Ch)_testing.ipynb

Main differences: all form of metric in 2) are considerably higher.
Given that code, data, seed at kept similar, can conclude that augmenting input with retrieved info improves performance. -> bringing in external info improves perf

<br />
### To reproduce results:
Change ./final_train.csv etc.., to ./dataset/... in the following codes in the training and testing file
![image](https://user-images.githubusercontent.com/54625060/170484339-21d1db66-19d9-4126-8bb3-5a2f8b684b7b.png)
It doesnt take long to train. In the training file, epoch can be changed to 3 instead of 10.
