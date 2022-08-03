# Fusing Retrieved Information With Classification
In this simple project, I augment inputs to a model (BERT) with their most relevant retrieved information before predicting their labels. 

This simple technique led to a 7% improvement vs the base line.

## Main Idea:
The goal of this classification is to label passages as fake/real via a Bert model.

I included an Information Retrieval (via BM25) step to retrieve relevant pieces (Top 3) of information from a knowledge base (external and changeable) pertaining to the passage that is to be classified.

Subsequently, the retrieved information is fused with the passage by prepending retrieved info to the input.
<br/>

## Overview of Solution:
![image](https://user-images.githubusercontent.com/54625060/170184988-c058e8ba-687c-40e4-9515-2f32c5881012.png) <br />

#### 1) Baseline without fusion
![image](https://user-images.githubusercontent.com/54625060/170638629-d9083dea-f623-4e59-9821-da4095d55d56.png)

<b>training file</b>: /src/training/submit_single_query_training.ipynb

<b>testing file</b>: /src/testing/submit_single_query(p)_testing.ipynb <br />
#### 2) With fusion
![image](https://user-images.githubusercontent.com/54625060/170638587-4f102fe5-c723-45df-b8f0-fc1c67faaec1.png)

<b>training file</b>: /src/training/submit_paired_query(p%2Ch)_training.ipynb

<b>testing file</b>: /src/testing/submit_paired_query(p%2Ch)_testing.ipynb

Main differences: all form of metric in 2) after applying fusion, are considerably higher than 1) (without).
Variables (less the retrieved info) are kept the same thus we can conclude that augmenting input with retrieved info does improve performance -> thereby bringing in external info can improve perf <br />

## Implications:
Why is this important? Because this experiment shows that we can augment model parameters with external knowledge to update the prediction. 

Although there may not be relevant information to be retrieved for a fake news, bringing in external information will still be beneficial in most cases as the model is very robust to noise.


