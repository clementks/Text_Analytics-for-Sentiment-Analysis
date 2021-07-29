# Text_Analytics-for-Sentiment-Analysis
Using Natural Language Processing to determine Customer Sentiments and Predict Customer Churn (Negative Review) or Retention (Positive Review) 

# Business Problems


   In this competitive E-commerce shopping experience, any firm's marketing team can realize the gap in their sales revenue and their e-commerce web content search engine optimization efforts & high customer churn-rate.  Their online store can be bombarded with customer’s reviews that is hardly harness in allowing marketing team to learn a great deal about customer sentiment and preferences towards products or services. Marketing team could realize their assortment planning based on customer purchase patterns but find it challenging to solicit & grasp customer’s opinions on particular products and their categories. Hence, they may want to unleash other avenues to enhance their existing content marketing development of each product category by leveraging on topic modeling for better content development in their landing page.
	 Hence, topic modelling & document clustering techniques can be employed to discover the topics with relation to each cluster of product class and department, to use natural language processing (NLP) for sentimental analytics to assess any potential customer churn and use it for to strengthen organic SEO in order to improve search engine page-ranking (SERP). Content marketing is the best way to improve organic search engine optimization by posting relevant images built from topic modelling and web document clustering. By creating high quality content with images that's focused around targeted keywords and phrases using techniques from topic modelling and document clustering, it can improve visibility on the search engines with higher leads acquisition to conversion rate.


**i.	  State the type of data that are required.**

   The most critical data for this project is the Review Text to identify key topic phrase, which is complemented by data from other fields to provide relevancy & association to the target audience.

![image](https://user-images.githubusercontent.com/32416129/127283175-5cc036cf-e101-44ab-8dda-8803f27303a7.png)


**ii.	  Provide relevant business and operational definitions of the data.**


   Product reviews are multifaceted, and hence the textual content of product reviews is an important determinant of consumers' choices as to how textual data can be used to learn consumers' relative preferences for different product features and also how text can be used for predictive modeling of future changes in sales. Both negative sentiment and positive sentiment of the reviews are individually significant predictors influence in predicting future sales or customer demand which is an important part of Supply Chain Management in managing finished goods inventory stockpoints.


Rating: Quantitative measure of customer’s satisfaction level of the purchase transaction with respect to the Clothing ID.

Feedback Count: Quantitative measure on the number of customers providing their feedback from the survey made after their purchase transaction.

Review Text: Voice of Customer (VoC) or qualitative information of customer’s satisfaction level with respect to the Clothing ID of the purchase transaction.

   Product Reviews can be useful to explore unpromising avenues in ways to improve web content for new visitors & ads re-targetting to serve online visitors who have already visited any e-tailers website before. 


**iii. Details for the prediction involved.**


   The typical prediction involved in the e-commerce space retailer or merchants comprises of customer retention and churn model based on implicit feedbacks such as past purchase transactions recency, frequency, monetary_value and explicit feedbacks such as text reviews and satisfaction rating level. For any e-tailers, this kind of prediction capability is also extremely important in order to manage the supply chain efficiently as well as ensure customer satisfaction, as such this project investigates the efficacy of various modeling techniques, namely, regression analysis, decision-tree analysis. The scope of this project include using text reviews from explicit feedbacks to predict satisfaction level or sentiments polarity expressed in categorical scale of “POSITIVE”, “NEUTRAL” and “NEGATIVE”.
	Satisfaction rating level extracted from the web mining process is firstly converted from ordinal scale of 1-5 into categorical format as “POSITIVE”, “NEUTRAL” and “NEGATIVE” before it is split into training and test data for predicting sentiments using various supervised, unsupervised and ensemble machine learning algorithm within classification model.
	All reviews with rating 1 and 2 are marked as ‘negative’ reviews, reviews with rating 3 is marked as ‘neutral’ and reviews with rating 4 and 5 are marked as ‘positive’.

![image](https://user-images.githubusercontent.com/32416129/127283489-6627d8ef-51b5-4a35-8980-e5e7d552146b.png)

   The machine learning models that will be evaluated include traditional supervised multinomial Naïve Bayes models, Support Vector Machines (SVM), unsupervised machine model include DecisionTree with gini criterion, and Ensemble model such as AdaBoost and RandomForest for predicting sentiment polarity.
   
   ![image](https://user-images.githubusercontent.com/32416129/127283662-58938f0d-a555-4e67-93aa-8fea9889e8fc.png)


   Leverage on pipeline module from Scikit-Learn to build this machine learning pipeline which combines CountVectorizer, TfidTransformer and various supervised, unsupervised and ensemble machine model(s) as mentioned earlier.
   
   ![image](https://user-images.githubusercontent.com/32416129/127285483-dba99633-0393-4fa9-a3da-951eacf3d383.png)
   
   GridSearchCV was also used for tuning the hyper-parameters of an estimator for each of the supervised, unsupervised and ensemble machine model.
   
   ![image](https://user-images.githubusercontent.com/32416129/127285658-d69e5b68-e7f1-4f90-ad71-83bfcff7546e.png)

	
**iv.	Text preprocessing – Understand and clean the text data.**


The Data pre-processing comprises of these steps:

![image](https://user-images.githubusercontent.com/32416129/127285875-a2c9ba6b-b1a6-4038-82f7-e3cfb575f3fc.png)


**v.	Feature engineering – Represent the unstructured text data in a structured format using various feature engineering models.**


   There are some potential problems that might arise with the Bag of Words model when it is used on large corpora of text data. Since BOW feature vectors are based on absolute term frequencies, there might be some terms that occur frequently across all documents and these may tend to overshadow other terms in the feature set. For words that don’t occur as frequently, it might be more interesting and effective to use combination of two metrics, term frequency (tf) and inverse document frequency (idf ) to identify as features relevant to specific business domain arena (ie: fashion clothing).
Inverse document frequency denoted by idf is the inverse of the document frequency for each term and is computed by dividing the total number of documents in our corpus by the document frequency for each term and then applying logarithmic scaling to the result.

Before fitting into the NMF & LDA model, term frequency-inverse document frequency (tf-idf) features was extracted from the data sets with the function call, TfidfVectorizer. This function converts a collection of documents to a matrix of tf-idf features.

   Depending on the size of the vocabulary that is required for business objectives in the respective departments (ie: Bottoms, Intimate, Top, Dresses etc.), TfidfVectorizer  min_df or max_df shall be adjusted accordingly. For this project, min_df was set to 0.07 and approximately 50 vocabulary words are generated.
While TF-IDF is effective methods for extracting features from text, due to the inherent nature of the corpus being just a bag of unstructured words. To ensure topic modelling stays relevant and precise for each product category with respect to department or even brand labels, the raw corpus data was sliced before applying TfidfVectorizer and LDA / NMF topic modelling. 

![image](https://user-images.githubusercontent.com/32416129/127287280-485ff689-f4e6-410e-9f6b-bb83303ddbb4.png)

![image](https://user-images.githubusercontent.com/32416129/127287428-ced24e64-5dbb-422b-9d7a-f57470acc5e8.png)

 LDA, or Latent Derelicht Analysis is a probabilistic model, and to obtain cluster assignments, it uses two probability values: P( word | topics) and P( topics | documents). These values are calculated based on an initial random assignment, after which they are repeated for each word in each document, to decide their topic assignment. 
 
 To build an LDA model, i wanted to find the optimal number of topics to be extracted from the caption dataset. I use the coherence score of the LDA model to identify the optimal number of topics. Approach to finding the optimal number of topics is to build many LDA models with different values of number of topics (k) and pick the one that gives the highest coherence value. Choosing a ‘k’ that marks the end of a rapid growth of topic coherence usually offers meaningful and interpretable topics. Picking an even higher value can sometimes provide more granular sub-topics.
 
 ![image](https://user-images.githubusercontent.com/32416129/127288025-c2516bcb-9f15-4e02-b032-6d2d1fd6427d.png)

When seeing the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large. The compute_coherence_values() (see below) trains multiple LDA models and provides the models and their corresponding coherence scores.


**The LatentDirichletAllocation (n_components = TOTAL_TOPICS) is manually refined from the number of optimal topics of 6 ** derived from highest coherence score to a smaller scale upon each observation to ensure there’s no repetition of topics terms. The reason is for reducing TOTAL_TOPICS to a minimal can be explained by any business instinct to find a niche topic that is unique and relevant.  To ensure topic terms are unique and relevant for each department could be different, hence the parameter settings for LatentDirichletAllocation(n_components =TOTAL_TOPICS) could vary from each dataframe associated with each department.


![image](https://user-images.githubusercontent.com/32416129/127288711-a8cfa4f5-c833-470d-9b68-24af79983e37.png)


# Sentiment Analysis

The results in this analysis confirms our previous data exploration analysis, where the data are very skewed to the positive reviews as shown by the higher support counts in the classification report. 
Despite of the fact that Neutral and Negative results are not very strong predictors in this data set, it still shows 83% accuracy level in predicting the sentiment analysis, which we tested and worked very well when inputting unseen text (Review_Test). 
Finally, the overall result from Confusion Matrix shows the performance of classification model (or “classifier”) for Linear Support Vector Model on a set of test data for which the true values are known.

![image](https://user-images.githubusercontent.com/32416129/127293239-6b7ed85c-bdf9-442d-906a-ba62e85aa4a3.png)

We see that Positive sentiment prediction can be daunting with neutral and negative ratings being misclassified with count of 70 and 33 respectively. However, based on the overall number of significant positive sentiment at a score 747, this misclassification error of 17% are considered significant depending on the application if this happens to be implemented in recommender system.
The evaluation in the section below describes the performance of SentiWordNet & VADER lexicon-based model.

![image](https://user-images.githubusercontent.com/32416129/127293559-6aa73c32-30a6-45b6-bcd4-acea59711189.png)

![image](https://user-images.githubusercontent.com/32416129/127293675-bf4ec6b4-f1a1-43d4-bf4d-8cec6993f309.png)


Based on the threshold settings applied the same to both models, the classification report shows that VADER has a better accuracy performance with lower probabilities of false negative and false positive. The confusion matrix table shows the count of true positive, true negative, false negative and false positive from the classification. Hence, VADER model will be preferred choice in predicting unseen observations. 

![image](https://user-images.githubusercontent.com/32416129/127293816-d12a6b96-2559-44ed-aff0-15ebb7ac7259.png)


**i.	Deploy the final chosen model to predict new unseen observations.**

For the Unsupervised lexicon-based model given that threshold value settings applied are the same., VADER lexicon-based model will be adopted instead of SentiWordNet due to higher overall accuracy.






 
