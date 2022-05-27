# simple-Bert-Retrieval-augmented-classification
Augmenting inputs to BERT with retrieved information, before predicting their labels

## Main Idea:
The goal of this classification is to label passages as fake/real via a Bert model.
What is done differently is that in this experiment (vs normal classification), I included an Information Retrieval (via BM25) step to retrieve relevant pieces of information (external and changeable) pertaining to the passage that is to be classified.
Subsequently, the retrieved information (top relevant/only one) is fused with the passage by prepending retrieved info to Bert input.
This form of fusion done is the simplest and is called 'Priming'. <br/>

## Overview of Solution:
![image](https://user-images.githubusercontent.com/54625060/170184988-c058e8ba-687c-40e4-9515-2f32c5881012.png) <br />

## Results
For the sake of reproducibility, the Prediction dataset, training code and seed(123) are kept same across the experiments, except the lines required to alter the fusion mechanism 

#### 1) Baseline without fusion
![image](https://user-images.githubusercontent.com/54625060/170638629-d9083dea-f623-4e59-9821-da4095d55d56.png)

<b>training file</b>: /src/training/submit_single_query_training.ipynb

<b>testing file</b>: /src/testing/submit_single_query(p)_testing.ipynb <br />
#### 2) With fusion
![image](https://user-images.githubusercontent.com/54625060/170638587-4f102fe5-c723-45df-b8f0-fc1c67faaec1.png)

<b>training file</b>: /src/training/submit_paired_query(p%2Ch)_training.ipynb

<b>testing file</b>: /src/testing/submit_paired_query(p%2Ch)_testing.ipynb

Main differences: all form of metric in 2) after applying fusion, are considerably higher than 1) (without).
Given that the code, dataset, seed are kept similar, can conclude that augmenting input with retrieved info improves performance, -> thereby bringing in external info improves perf<br />

## To reproduce results:
Change ./final_train.csv etc.., to ./dataset/... in the following codes in the training and testing file
![image](https://user-images.githubusercontent.com/54625060/170484339-21d1db66-19d9-4126-8bb3-5a2f8b684b7b.png)


It doesnt take long to train. In the training file, epoch can be changed to 3 instead of 10.

## References:
Based on survey: https://openreview.net/forum?id=9_oCNR6R9l2

Cited as:

    @article{yuntian2022bertkbfusion,
      title   = "Bert-IR-augmented-classification",
      author  = "Yun Tian, Gan",
      journal = "",
      year    = "2022",
      url     = "https://github.com/gYunTian/simple-Bert-IR-augmented-classification"
    }
