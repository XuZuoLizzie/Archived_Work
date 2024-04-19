# Archived Work
## Overview
This repository archived a collection of biomedical data science work from Xu Zuo, Ph.D. in Biomedical Informatics at UTHealth Houston.

## Catalogs
**Colab Notebooks**

The directory archived a set of Jupyter Notebooks for a varity of tasks in biomedical data science. All notebooks can be imported and run in Google Colab.

- Bio Entity Extration: The notebook introduces a demo that uses data from Europe PMC Annotations to train an NLP pipeline that extracts Cell entities from full-text articles.

- Heart Failture Prediction: The notebook implements and tests a series of deep learning models for predict diabetic patients risk to develop heart failure using their available electronic health records (EHRs) data.

- Retina Image Analysis: The notebook implements a pretrained inception v3 model as a base and retraining some modified final layers with attention. The goal is to train a retinopathy model for retina image analysis. 

- PubMed Abstract Generation: The notebook trains and inferences a character-level RNN model to generate PubMed abstracts.

**Biomedical Data Mining**

The directory archived a series of projects in biomedical data mining. The analysis scripts are all in Python. Before running the analysis, nevigate to the sub-directory, create a virtual environment and install essential packages using `pip install -r requirements.txt`. The analysis can be executed using `python analysis.py`.

- Supervised Classification on EHR: This directory implements algorithms of supervised classification and unsupervised clustering for biomedical text data.

- Tabular Data Analysis: This directory includes modules that preprocess and summarize tabular EHR data.

- Text Classification & Clustering: This directory implements algorithms of supervised classification and unsupervised clustering for structured EHR data.


- Using FHIR API: This directory includes modules that load and preprocess EHR data from FHIR API.

- XML Data Processing: This directory includes modules that preprocess and summarize EHR data in XML format.


**Preprint Classification**

The directory includes source codes and documentations for COVID-19 preprint classification. The goal of this project is to assist literature screening by developing an automated approach to identify the study type (“RCT”, “observational study”, or “other”) of studies related to COVID-19 treatments. In this task, both machine learning-based and transformer-based approaches are implemented and evaluated. The classifcation pipeline can be run using either the Python script or Docker container.

**FindU**

The direcotry includes the source codes of FindU, a web application that helps students to search and find universities based on their preferences. The web application is build using MangoDB and Django. 


**Statstical Analysis**

This direcotry includes reports for the course Intermediate Biostatistics (2212PHW-1700L700) at UTHealth. All analysis were carried out in STATA.