# simple-Bert-Retrieval-augmented-classification
Augmenting inputs to BERT, with retrieved information, before predicting their labels

### Main Idea:
The goal of this classification is to label passages as fake/real via a Bert model.
What is done differently is that in this experiment (vs normal classification), included Information Retrieval (via BM25) step to retrieve relevant pieces of information pertaining to the passage that is to be classified.
Subsequently, the retrieved information (only one) is/are fused with the passage by prepending retrieved info to Bert input.
This form of fusion done is the simplest and is called 'Priming'.

### Overview of Solution:
![image](https://user-images.githubusercontent.com/54625060/170184988-c058e8ba-687c-40e4-9515-2f32c5881012.png)

### Results
For the sake of reproducibility, the Prediction dataset, training code and seed(123) are kept same across the experiments, except the lines required to alter the fusion mechanism 
#### 1) Baseline without fusion
![single query](https://user-images.githubusercontent.com/54625060/170483673-9fcb49da-2ce5-4453-b33e-fdb63479a454.jpg)

<b>training file</b>: /src/training/submit_single_query_training.ipynb

<b>testing file</b>: /src/testing/submit_single_query(p)_testing.ipynb

#### 2) With fusion
![paired_query](https://user-images.githubusercontent.com/54625060/170483836-9fb4bd50-72db-4bdf-9dcb-b808f0c9340e.jpg)

<b>training file</b>: /src/training/submit_paired_query(p%2Ch)_training.ipynb

<b>testing file</b>: /src/testing/submit_paired_query(p%2Ch)_testing.ipynb

Main differences: all form of metric are considerably higher.
Given that code, data, seed at kept similar, conclude that augmenting input with retrieved info improves performance, and can bring in external info as well

### To reproduce results:
Change ./final_train.csv etc.., to ./dataset/... in the following codes in the training and testing file
![image](https://user-images.githubusercontent.com/54625060/170484339-21d1db66-19d9-4126-8bb3-5a2f8b684b7b.png)
