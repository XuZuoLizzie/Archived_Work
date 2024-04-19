# Xu Zuo - Text classification for COVID-19 treatment preprints
## Background
Since the COVID-19 pandemic began, researchers have been investigating effective treatments for the coronavirus. Systematic reviews and meta-analysis of clinical studies are useful approaches to analyze results from a number of relevant studies, thus increasing the statistical ability to detect the true value of treatments. However, manual search and extraction of biomedical literature can be both time consuming and laborious. Therefore, the goal of this project is to assist literature screening by developing an automated approach to identify the study type of studies related to COVID-19 treatments.
## Task
This is a multi-label classification task. The first step is data preprocessing, which includes data parsing and feature extraction. Basic features include word shapes, prefix-suffix, and bigrams. Additional features include TD-IDF and word embedding. For the label prediction I plan to use three machine learning models (logistic regression, XGBoost, and support vector machines) as the baselines. The baseline models will be implemented using the scikit-learn library. Additionally, I plan to test two deep learning models (FastText and TextCNN) on this task. 
## Dataset
The preprints used in this project were pre-labeled as “RCT”, “observational study”, and “other”. I downloaded preprint data, including titles, author names, affiliations, release dates, and abstracts from preprint servers (MedRxiv and BioRxiv). The preprint datasets are saved as JSON files. In case preprint datasets are inadequate for training and testing deep learning models, I also plan to retrieve and label additional PubMed articles related to COVID-19 treatments. The PubMed article datasets will be saved as XML files.
## Resutls and evaluation
The final deliverables will be the predictive labels of test data. The results will be reported numerically using evaluation metrics. I will also demonstrate results visually by plotting the receiver operating characteristic (ROC) curve. The results reported will be based on 10-fold cross validations.
## Running the pipeline
Please follow the instructions to reproduce the analysis results.

**Set up the environment**

The pipeline requires Python 3.7.
You can run the analysis pipeline with baseline models on local desktops. 
If you would like to run with the deep learnin model.
Please make sure you have GPUs available. 

**Clone repository**

`git clone https://gitlab.com/bmi6319/2021-spring/xu-zuo.git`

**Prediction with baseline model**

Install required packages:

`pip install -r requirements.txt`

Navigate to the `model` directory:

`cd model/`

Run the following script:

`python3 lr_classification.py`

The evaluation results will be printed onto your screen.

**Run the pipeline with customized datasets**

Note: The pipeline is only applicable to multi-class classification tasks.

If you would like to run this pipeline using your own datasets. 
Please convert your data into a TSV file.
Please refer to the sample dataset `preprint_classification_data.tsv` in `data` directory.
In the TSV file, there should be at least two columns: one named `text`, which stores plain text; Another named `study type`, which stores the labels.
