# Text_Analytics-for-Sentiment-Analysis
Using Natural Language Processing to determine Customer Sentiments and Predict Customer Churn (Negative Review) or Retention (Positive Review) based on the statistical skewness

**Business Problems**


   In this competitive E-commerce shopping experience, any firm's marketing team can realize the gap in their sales revenue and their e-commerce web content search engine optimization efforts & high customer churn-rate.  Their online store can be bombarded with customer’s reviews that is hardly harness in allowing marketing team to learn a great deal about customer sentiment and preferences towards products or services. Marketing team could realize their assortment planning based on customer purchase patterns but find it challenging to solicit & grasp customer’s opinions on particular products and their categories. Hence, they may want to unleash other avenues to enhance their existing content marketing development of each product category by leveraging on topic modeling for better content development in their landing page.
	Hence, topic modelling & document clustering techniques can be employed to discover the topics with relation to each cluster of product class and department, to use natural language processing (NLP) for sentimental analytics to assess any potential customer churn and use it for to strengthen organic SEO in order to improve search engine page-ranking (SERP). Content marketing is the best way to improve organic search engine optimization by posting relevant images built from topic modelling and web document clustering. By creating high quality content with images that's focused around targeted keywords and phrases using techniques from topic modelling and document clustering, it can improve visibility on the search engines with higher leads acquisition to conversion rate.


**i.	State the type of data that are required.**

   The most critical data for this project is the Review Text to identify key topic phrase, which is complemented by data from other fields to provide relevancy & association to the target audience.

![image](https://user-images.githubusercontent.com/32416129/127283175-5cc036cf-e101-44ab-8dda-8803f27303a7.png)


**ii.	Provide relevant business and operational definitions of the data.**


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
   
	
**iv.	Text preprocessing – Understand and clean the text data.**


The Data pre-processing comprises of these steps:
	Tokenization: Split the text into words. Lowercase all words so that it aggregate raw text corpus into similar or common words which is useful when processing the subsequent stopword list and spelling-checker.
	Words that have fewer than 2 characters are removed since it does not provide much value for topic modelling or features term.
	Build stopwords list to remove common vocabulary and English words that does not provide any value for features engineering related to context of woman clothing or fashion.
	Use SpellChecker instead of Textblob. Spellchecker has been tested and verified by myself that it has better & higher accuracy than Textblob which often mis-corrected. Spelling-checker is important to correct and remove unwanted errors. 
	Remove any digit or numeric characters since it has little value or relevance to topic modelling & bring any insights to marketing business context.
	Remove html tag characters since it has little value or relevance to topic modelling & bring any insights to marketing business context.  HTML tags, which do not add much value when analyzing text for topic modelling, hence this function help to remove such unnecessary content.
	Removing accented characters: In text corpus that is dealt with in this woman’s clothing review in English Language, there are accented characters/letters that needs to be converted and standardized into ASCII characters. A simple example is converting é to e



c.	Feature engineering – Represent the unstructured text data in a structured format using various feature engineering models.


   There are some potential problems that might arise with the Bag of Words model when it is used on large corpora of text data. Since BOW feature vectors are based on absolute term frequencies, there might be some terms that occur frequently across all documents and these may tend to overshadow other terms in the feature set. For words that don’t occur as frequently, it might be more interesting and effective to use combination of two metrics, term frequency (tf) and inverse document frequency (idf ) to identify as features relevant to specific business domain arena (ie: fashion clothing).
Inverse document frequency denoted by idf is the inverse of the document frequency for each term and is computed by dividing the total number of documents in our corpus by the document frequency for each term and then applying logarithmic scaling to the result.

Before fitting into the NMF & LDA model, term frequency-inverse document frequency (tf-idf) features was extracted from the data sets with the function call, TfidfVectorizer. This function converts a collection of documents to a matrix of tf-idf features.

# Build feature matrix for a unigram for a cleaned normalized corpus

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='word', min_df=0.1, max_df=0.8, smooth_idf=True)
tfidf_features = vectorizer.fit_transform(wc_reviews_cleaned['cleaned_words'])
vocabulary = vectorizer.get_feature_names()
print(tfidf_features.shape)
print(vectorizer.get_feature_names())

 
(22628, 17)
['beautiful', 'color', 'comfortable', 'cute', 'dress', 'fabric', 'great', 'large', 'length', 'material', 'nice', 'perfect', 'size', 'small', 'soft', 'top', 'wear']

# Build feature matrix for a tri-gram after topic modelling

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', min_df=0.1, max_df=1.0, smooth_idf=True, ngram_range=(1,3))
tfidf_features = vectorizer.fit_transform(wc_reviews_topic['Topic'])
                                            
# get feature names
feature_names = vectorizer.get_feature_names()

print(tfidf_features.shape)
# print sample features
print(feature_names[:20]) 



When TfidfVectorizer  min_df is increase to 0.2, the size or length of the feature_names is reduced from 17 to 5. Hence, this project will have min_df=0.1



 


While TF-IDF are effective methods for extracting features from text, due to the inherent nature of the model being just a bag of unstructured words.


d.	Actual text analysis using machine learning:

i.	Train models using various supervised learning, unsupervised learning, reinforcement learning, and deep learning algorithms.




LDA, or Latent Derelicht Analysis is a probabilistic model, and to obtain cluster assignments, it uses two probability values: P( word | topics) and P( topics | documents). These values are calculated based on an initial random assignment, after which they are repeated for each word in each document, to decide their topic assignment. In an iterative procedure, these probabilities are calculated multiple times, until the convergence of the algorithm

from sklearn.decomposition import LatentDirichletAllocation

lda_model = LatentDirichletAllocation(n_components =TOTAL_TOPICS, max_iter=500, max_doc_update_iter=50, doc_topic_prior=None,learning_method='online', batch_size=128, learning_offset=50., topic_word_prior=None, random_state=42, n_jobs=16)
lda_document_topics = lda_model.fit_transform(tfidf_features)



lda_topic_terms = lda_model.components_

 

 

Non-negative Matrix Factorization is a Linear-algebraic model, that factors high-dimensional vectors into a low-dimensionality representation. Similar to Principal component analysis (PCA), NMF takes advantage of the fact that the vectors are non-negative. By factoring them into the lower-dimensional form, NMF forces the coefficients to also be non-negative.
Given the original matrix A, we can obtain two matrices W and H, such that A= WH. NMF has an inherent clustering property, such that W and H represent the following information about A:
A (Document-word matrix) — input that contains which words appear in which documents.
W (Basis vectors) — the topics (clusters) discovered from the documents.
H (Coefficient matrix) — the membership weights for the topics in each document.
W and H is being calculated by optimizing over an objective function (like the EM algorithm), updating both W and H iteratively until convergence.


 


Document clustering leverage using these distance similarity (ie: cosine similarity) to identify how similar a text document is to any other document(s) using the topics terms generated from LDA. There are several similarity and distance metrics that are used to compute document similarity which include cosine distance/similarity, Euclidean distance, manhattan distance, BM25 similarity, jaccard distance, and so on. In our analysis, we use perhaps the most popular and widely used similarity metrics—cosine similarity and compare pairwise document similarity—based on their TF-IDF feature vectors.

The cosine similarity is used to calculate the linkage matrix before it is used to develop hierarchical structure as a dendrogram.


To work with Ward Clustering Algorithm, the following steps were performed:
•	Prepare a cosine distance matrix to calculate ward linkage_matrix
•	Ward’s minimum variance method is used as our linkage criterion to minimize total within-cluster variance
•	Plot the hierarchical structure as a dendrogram.


### Calculate Linkage Matrix using Cosine Similarity


def ward_hierarchical_clustering(feature_matrix):
    
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix

### Plot Hierarchical Structure as a Dendrogram


def plot_hierarchical_clusters(linkage_matrix, data, figure_size=(8,12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    wc_review_topic_clusters = data['Topic'].values.tolist()
    # plot dendrogram
    ax = dendrogram(linkage_matrix, orientation="left", labels=wc_review_topic_clusters)
    plt.tick_params(axis= 'x',   
                    which='both',  
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig(path + 'ward_hierachical_clusters.png', dpi=300)






















Below is the hierarchical dendrogram from LDA modelling topic
 


We can clearly see from Affinity Propagation algorithm, it has identified the four distinct categories in our topic modelled documents with cluster labels assigned to them. This should give us a good idea of how we can cluster topic modelled documents 




 

	Document-level embeddings using Word2vec

Once features for each document are established using TD-IDF, cluster these documents using the affinity propagation algorithm, which is a clustering algorithm based on the concept of “message passing” between data points. 
It does not need the number of clusters as an explicit input, which is often required by partition-based clustering algorithms such as K-means. The algorithm has clustered each document into the right group based on our Word2Vec features. Pretty neat! We can also visualize how each document is positioned in each cluster by using Principal Component Analysis (PCA) to reduce the feature dimensions to 2D and then visualizing them (by color coding each cluster).



from gensim.models import word2vec
# Set values for various parameters
feature_size = 200 # Word vector dimensionality
window_context = 3 # Context window size
min_word_count = 250 # Minimum word count
sample = 1e-3 # Downsample setting for frequent words

wpt = nltk.WordPunctTokenizer()
tokenized_corpus = wc_reviews_cleaned['cleaned_words'].apply(word_tokenize)
w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, window=window_context, min_count = min_word_count, sample=sample, iter=50)

 
 

ii.	Evaluate the performance of the models.

	
	Topic coherence can be defined as how interpretable a topic is based on the degree of relevance between the words within the topic itself. The topic coherence measures used in this assignment aims to evaluate the quality of topics from a human-like perspective. However, it has been found that topic coherence methods are not always reflective of how a human views the topics. Consequently, other metrics based on word co-occurrences and a Mutual Information Approach which are more representative of how a human would approach topic evaluation. It is for this reason word co-occurrence and the mutual information approach is used in this work.


	Although there is certainly a fair degree of subjectivity involved when judging interpretability of topics, we believe that it’s fair to conclude that NMF generates more interpretable topics overall.
The one possible explanation for this difference is that NMF performs better on a smaller number of documents than LDA.
Using coherence score as the evaluation measure, NMF and LDA is pretty comparable. One change that we found quite necessary to make, was to include in the list of stop words common English terms which has no relevance to key phrase or topics. After building the stop words list, the result are left with much more interpretable topics. 
	Non-negative Matrix Factorisation (NMF) based method however, does not directly deal with improving coherence but NMF has a tendency to be more unstable than probabilistic methods like LDA. In that sense, stability almost equates to model reproducibility. Taking into account on topic easy for human evaluations, LDA is the final chosen topic model.


 
