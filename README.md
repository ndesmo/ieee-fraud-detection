# ieee-fraud-detection
Detect fraudulent online transactions

In this project I am attempting to correctly classify fraudulent online e-commerce transactions, using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset.

In order to run the code, you must first download the input files from Kaggle. You can then refer to these from the command line when running either `process_data.py`, `train_classifier.py` or `run_model.py`


## Summary of analysis

Please see [Capstone Project.ipynb](https://github.com/ndesmo/ieee-fraud-detection/blob/master/libs/ieee_fraud_detection/notebooks/Capstone%20Project.ipynb) for a full detailed write up about the project.

In summary, I learned a great deal about data science pipelines in this project, and was able to achieve an AUC score of 0.86338 on the training dataset. Excited about my result, I ran the model on the training data and submitted my results to Kaggle where I only received a score of 0.750277. I have discussed potential avenues for improvement.

## Libraries required

It is recommended to use Anaconda, so that the following libraries will be simple to install or already installed:

* pandas
* numpy
* sklearn
* sqlalchemy

Also, the code refers to itself. Please add the ieee_fraud_detection directory to your PYTHONPATH.

## License

This project is published under the Apache 2.0 Open Source License
http://www.apache.org/licenses/LICENSE-2.0

## Acknowledegments

I have used material from the following page to assist in my learning about how to code up ML pipelines: https://gist.github.com/amberjrivera/8c5c145516f5a2e894681e16a8095b5c

This project is submitted as part of the Data Scientist Nanodegree at Udacity, which I would highly recommend.
