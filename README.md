# Neutral Review Classification

## Business Case
Consumer reviews has historically been one of the best ways for business owners to understand more fully the needs and wants of their consumers, thus being an important tool for businesses to leverage in order to create a complete and enjoyable experience that encompasses their consuemr base. One important tool for this is business intelligence. However, business intelligence suites are generally expensive and inflexible, making it difficult for smaller businesses that are not largely scaled to gain a sufficient amount of value from these business suites to justify spending the amount it costs to utilize them. Thus, I wanted to create a business intelligence tool that could add value to consumer reviews that may not seem informationally valuable at first glance. 

The Neutral Feedback Problem: generally speaking, 4 and 5 star reviews are positive consumer experiences, and the language in these reviews reflect that with highlights and recommendations to others. On the other end of the spectrum, 1 and 2 star reviews are generally negative consumer experiences, and the language in these reviews are facets of the experience that consumers think should be improved. This can more easily be thought of as: 4-5 star reviews - highlights, and 1-2 star reviews - improvements needed. However, in the middle ground, a place of difficult interpretability, are the neutral 3 star reviews. These reviews typically have a middle ground between "highlights" and "improvements needed", thus making it difficult to quickly glean any information, unless a human manually goes through and classifies the language in the context of the review. 

Goal: the goal of this project is to add informational value to the difficult-to-interpret reviews by classifying them as positive or negative, hopefully offering a quick way for small business owners to gain value from these neutral reviews. 

## Data 
I used a Yelp Reviews Dataset from Kaggle (https://www.kaggle.com/luisfredgs/yelp-reviews-csv)

The dataset contained **5,261,668 reviews**, but the majority of this project was modeled on a randomly sampled subset of 500,000 reviews for computational reasons

Star Rating Distribution: 
[images/stars_distribution.png]

5 stars: 2,253,347 reviews
4 stars: 1,223,316 reviews
3 stars: 615,481 reviews
2 stars: 438,161 reviews
1 star: 731,363 reviews

## Data Preparation
I used TF-IDF Vectorization for the majority of the modeling. I have plans to use tokenization as well, as soon as it starts working computationally

## Modeling
Metric: F1 Score is pretty helpful here

Initially, I used three baseline models to see which would benefit most from hyperparameter tuning: 

[images/baseline_models.png]

I went with logistic regression as my model of choice, and after tuning with Grid Search, the test F1 for non-neutral reviews improved to 0.971. 

### Neutral Review Classification Process

I manually went through the neutral reviews in the dataset, classifying them as positive or negative experiences. What ended up being used was aproximately 150 neutral reviews in the training set (I know, huge class imbalance, but I'm working on it!) and about 50 neutral reviews as a separate validation set. 

Neutral Reviews Baseline vs. Final Model: 
Baseline Model: Validation F1 = 0.680
Final Model: Validation F1 = 0.687
