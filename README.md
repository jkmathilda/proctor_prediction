# LEIGS2026: Proctor Prediction Challenge
Welcome to the official "Proctor Prediction Challenge" as part of the 2nd Leipzig Geotechnical Symposium (LeiGS 2026)!
Digitalization is transforming geotechnical engineering at a rapid pace. To make optimal use of growing data volumes, the ability to combine modern data science methods with profound geotechnical expertise is more in demand today than ever before. Our challenge sits exactly at this intersection.
We are providing all participants with an exclusive dataset from the soil mechanics research laboratory of HTWK Leipzig.

## Task
Your task, either alone or in a team, is to develop a statistical model (e.g., based on Machine Learning). Based on classic soil properties (such as grain size distributions, consistency limits, etc.), this model should predict the compaction behavior of various soils in the Proctor test as accurately as possible.
Follow the Live Leaderboard in real-time to see how accurate your predictions are compared to the rest of the participants.

## The Ultimate Test
The true challenge comes at the end of the competition: The winning model will be evaluated on a separate, previously completely unknown test dataset. The winner is not the approach that best reproduces the known data (overfitting), but rather the model that demonstrates the highest generalization ability and precision on entirely new soil samples—just like in geotechnical engineering practice.

## Objective
The goal of the Challenge is to predict the maximum dry density ρPr and the corresponding optimum water content wopt from classification properties of the same soil sample. It is known that there is a strong correlation between the compaction parameters (ρPr, wopt) and the classification properties of soils determined in the laboratory. Parameters derived from the grain size distribution and the consistency limits are particularly relevant predictors. In geotechnical literature, numerous empirical formulations exist, but these are based on very different soil groups and test boundary conditions.

The task of this competition is to build a robust, data-driven bridge between the classifying parameters and the actual compaction parameters for fine-grained, mixed-grained, as well as coarse-grained soils.

For this purpose, we are providing a dataset. This dataset should be used to develop a modeling approach that provides precise estimates for the following target variables:

Proctor density (maximum dry density): ρPr
Optimum water content: wopt

<img width="150" height="150" alt="Image" src="https://github.com/user-attachments/assets/158f0168-922e-4f70-8311-446b5d4d69d1" />
Figure 2: Proctor curve based on 5 individual tests under varying water content with constant compaction energy. A polynomial function is used as a fitting function to determine the maximum dry density ρPr and the associated optimum water content wopt.

<img width="120" height="80" alt="Image" src="https://github.com/user-attachments/assets/8b06e853-b620-4ce4-8b99-92afab2f0283" />
<img width="120" height="80" alt="Image" src="https://github.com/user-attachments/assets/f838887a-22ff-4d3d-b171-35f83d594913" />
Figure 3: The grain size distribution (left) describes the composition of soils according to the percentage mass fractions per grain size. Soils with predominantly fine-grained components can be described in terms of their plasticity using consistency limits (right). Both tests provide fundamental insights into compactability.

<img width="110" height="80" alt="Image" src="https://github.com/user-attachments/assets/49d4a3cf-5624-49eb-ad1b-1ff52fa7d0e5" />
Figure 4: Influence of loss on ignition on the location of the Proctor optimum.

### Process Information
The developed model must meet the requirements for generalizability and robustness. To develop a model and have it evaluated in the context of the competition, it is best to follow these steps:
Data Exploration: Familiarize yourself with the data using the accompanying Dataset Description. Also, check the "Tutorials & Hints" section further down this page for a deeper understanding (videos of the test procedures, literature references, and a virtual lab tour).
Baseline Modeling: Follow the steps in the Python Getting Started Notebook. This will allow you to develop a first functioning model based on the training dataset in just a few minutes, perform an internal validation, and generate initial predictions for the test dataset.

Submission File Format
Submissions must be made as a CSV file. It must have a header row, and the id column must exactly match the row IDs of the test.csv file. A template is available for download in the Data Explorer.

'''
id,proctor_owc_pct,proctor_mdd_g_cm3
201,10.09,1.972
202,11.25,1.872
....
287,13.54,1.842
'''

## Data Description
The complete dataset comprises 288 observations (table rows), each characterized by 24 attributes (features – table columns). Each observation aggregates the derived parameters and test results of a specific soil sample, based on a series of geotechnical laboratory tests.
The goal is to predict the compaction parameters from the Proctor test (target variables) based on the remaining 22 features, which originate from classification and hydraulic laboratory tests.
The dataset was divided into two groups:
Training dataset (train.csv): Comprises 70 % of the data.
Test dataset (test.csv): Comprises 30 % of the data (internally split into approx. 15 % Public and 15 % Private).
Model Development (Training data train.csv) The actual compaction parameters are provided here. Your models will use this data to learn the relationships between the target variables and the 22 laboratory parameters. Feel free to expand the dataset with physically meaningful features as part of your feature engineering.
Model Evaluation (Test data test.csv) The target variables are intentionally hidden here – predicting them is your core task to test the generalizability of your models. The evaluation of the predicted parameters takes place in two stages via a defined Submission File using the leaderboard:
- Public Leaderboard (15 % of the data): Provides real-time feedback on model accuracy during the ongoing competition.
- Private Leaderboard (15 % of the data): Remains strictly secret until the end and determines the final ranking.

