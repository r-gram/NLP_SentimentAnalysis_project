# Evaluaing Models for Sentiment Analysis
## _Testing Models on Varying Data Sets_

## Motivation
Sentiment analysis has been used in the E-commerce space for many applications. It has been both a field of research with many methods being explored and an implemented system that businesses can use to make decisions about their products. This is especially important for businesses as public sentiment can influence how a product performs. If there is positive sentiment then the business may want to keep producing, or produce more of, said product. If there is negative sentiment then it may not be worth making as much of the product, or a different strategy to sell it might be needed. Either way, identifying the sentiment of the public is key and looking at reviews is extremely useful in doing this.
## Methods
There have been many methods over the years of accomplishing this. Typically, sentiment analysis has been used to classify text into three categories: positive, negative, and neutral. To accomplish this many machine learning techniques have been used such as decision trees, support vector machine, and more recently neural networks.

All these takes in tokenized data and then label each review as positive, negative, or neutral. Tokenized data is just the process of taking out all the “unnecessary” words from reviews. This includes words that do not directly affect the sentiment like “the”, “and”, and “this”. Some studies have shown the importance of domain-specific data for both choice and performance of model. The choice of model is important for both accuracy and time complexity. Some models may perform similarly but one finishes in minutes while the other takes half a day to run. Our contributions are twofold: Demonstrate how domain-specific data affects popular models in the field and compare those models on multiple data sets. The models being tested are as follows:
- Recurrent Neural Network w/ Long Short-Term Memory (RNN w/ LSTM)
- Convolutional Neural Network (CNN)
- Bidirectional Encoder Representations from Transformers (BERT)

To accomplish this, we will use three data sets that differ in their specificity. The least specific data set we look at is one that has a random selection of tweets that have some sort of sentiment. Next is a data set of reviews from Flipkart which sells a variety of items. The last data set we look at is one of women’s clothing reviews in the E-commerce space.
![Data Set Sizes](https://github.com/r-gram/NLP_SentimentAnalysis_project/blob/main/Data/images/SA_DataSetSizes.png?raw=true)
## Results
The RNN with LSTM model was trained using 5 epochs and had training accuracies of 93%, 92%, and 79% for each of the three datasets (Women’s E-Com, Flipkart, Twitter), respectively. Those models also achieved test accuracies of 90%, 89%, and 30%, respectively. This showed that the more domain specific data was important when training a RNN with LSTM model. 

The CNN model was trained using 5 epochs and resulted in training accuracies of 99%, 94%, and 80% for each of the three datasets, respectively. Those models also showed test accuracies of 91%, 88%, and 45%, respectively. This showed that the more domain specific data was important when training a CNN model. We can also see a 50% increase in test accuracy results between RNN and CNN models trained on the Twitter dataset.  

The BERT model was trained using 2 epochs and resulted in training accuracies of 96%, 95%, and 89% for each of the three datasets, respectively. Those models also showed test accuracies of 95%, 93%, and 82%, respectively. This showed that the more domain specific data was not nearly as important when training a BERT model. This substantial increase in test accuracy for the model trained on Twitter data suggested that the BERT model could potentially achieve high test accuracies regardless the specificity of the training data. 

##### Model Test Results

| Model | Epochs | Train Accuracy | Validation Accuracy | Test Accuracy 
| ----- | ------ | -------------- | ------------------- | ------------- 
| Women’s Clothing RNN |	5 |	0.93 |	0.93 |	0.90 
| Flipkart RNN |	5 |	0.92 |	0.91 |	0.89 
| Twitter RNN |	5 |	0.79 |	0.78 |	0.30 
| Women’s Clothing CNN |	5 |	0.99 |	0.93 |	0.91 
| Flipkart CNN |	5 |	0.94 |	0.90 |	0.88 
| Twitter CNN |	5 |	0.80 |	0.76 |	0.45 
| Women’s Clothing BERT |	2 |	0.96 |	0.95 |	0.95 
| Flipkart BERT |	2 |	0.95 |	0.94 |	0.93 
| Twitter BERT |	2 |	0.89 |	0.84 |	0.82 


## How to Run
All code is in an easy-to-use Jupyter Notebook. To run, download the [Notebook](https://github.com/r-gram/NLP_SentimentAnalysis_project/blob/main/Code/SentimentAnalysis_Project_Implementation.ipynb) and insure that all necessary packages are downloaded. 
Packages Needed:
- [Natural Language Processing Toolkit](https://www.nltk.org/install.html)
- [Matplotlib](https://matplotlib.org/stable/users/installing/index.html)
- [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)
- [NumPy](https://numpy.org/install/)
- [TensorFlow](https://www.tensorflow.org/install/pip)
- [SciKit-Learn](https://scikit-learn.org/stable/install.html)

The data used to train the models can be found [here](https://github.com/r-gram/NLP_SentimentAnalysis_project/tree/main/Data).

## Contributors
[Robert Gramlich](https://www.linkedin.com/in/robert-gramlich/)
[Nathan Thomas](https://github.com/NThom123)

## Citations
[1]   Dang NC, Moreno-García MN, De la Prieta F. Sentiment    Analysis Based on Deep Learning: A Comparative Study. Electronics. 2020; 9(3):483. https://doi.org/10.3390/electronics9030483
[2]   Dang NC, Moreno-García MN, De la Prieta F. Sentiment Analysis Based on Deep Learning: A Comparative Study. Figure 2. Electronics. 2020; 9(3):483. https://doi.org/10.3390/electronics9030483
[3]   Dang NC, Moreno-García MN, De la Prieta F. Sentiment Analysis Based on Deep Learning: A Comparative Study. Figure 3. Electronics. 2020; 9(3):483. https://doi.org/10.3390/electronics9030483
[4]   Dang NC, Moreno-García MN, De la Prieta F. Sentiment Analysis Based on Deep Learning: A Comparative Study. Figure 4. Electronics. 2020; 9(3):483. https://doi.org/10.3390/electronics9030483
[5]   Agarap, A. F. Statistical analysis on e-commerce reviews, with sentiment classification using bidirectional neural network (RNN) Electronics. 2020; https://arxiv.org/abs/1805.03687
[6]   Nick Brooks. 2018. Women’s E-Commerce Clothing Reviews. (2018). https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews 
[7]   Karimi, Rossi, Prati. Improving BERT Performance for Aspect-Based Sentiment Analysis Electronics. 2021; https://arxiv.org/abs/2010.11731
[8]   Karimi, Rossi, Prati. Improving BERT Performance for Aspect-Based Sentiment Analysis. Figure 2. Electronics. 2021; https://arxiv.org/abs/2010.11731
[9]   Karimi, Rossi, Prati. Improving BERT Performance for Aspect-Based Sentiment Analysis Figure 3. Electronics. 2021; https://arxiv.org/abs/2010
[10]    Brownlee, Jason. “Deep Convolutional Neural Network for Sentiment Analysis (Text Classification).” MachineLearningMastery.com, 2 Sept. 2020, https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/. 
[11]    prakharrathi25. “Sentiment Analysis Using Bert.” Kaggle, Kaggle, 24 Dec. 2020, https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert.  
