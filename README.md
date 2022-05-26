# simple-Bert-Retrieval-augmented-classification
Augmenting inputs to BERT, with retrieved information to predict labels

### Main Idea:
The goal of this classification is to label passages as fake/real via a Bert model.
We added a step on Information Retrieval (via BM25) to retrieve relevant pieces of information pertaining to the passage that is to be classified.
The retrieved information are then fused with the passages via 'Priming'.
The form of fusion done is is the simplest, by prepending retrieved passages to the input.

### Overview of Solution:
![image](https://user-images.githubusercontent.com/54625060/170184988-c058e8ba-687c-40e4-9515-2f32c5881012.png)

### Results
Prediction dataset, code and seed are kept same across the experiments 
#### Baseline without fusion
![image](https://user-images.githubusercontent.com/54625060/170185440-bfbb658b-fdae-42ad-b3e3-f728985dc12f.png)

#### With two different fusion
![image](https://user-images.githubusercontent.com/54625060/170185502-b1c896fa-e6ca-4a87-bcda-5c9b6f142ec2.png)

![image](https://user-images.githubusercontent.com/54625060/170185512-7b0e800e-e1ad-487a-b0d6-bd7ee086a96d.png)

Main differences: recall is higher (0.93 vs 0.92) and (0.97 vs 0.96)
