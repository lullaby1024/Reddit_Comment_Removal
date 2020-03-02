# Reddit_Comment_Removal
## Introduction
This repository contains source codes and analyses for the [Reddit Comment Removal task](https://www.kaggle.com/areeves87/rscience-popular-comment-removal). The overall goal is to identify features that are most predictive of comments getting removed, so as to automate comment removal process.

## Task 1: Bag of Words And Simple Features
- Set-up
  - `x`: comments
  - `y`: binary feature, with `1` indicating removal of comments and `0` otherwise.
  - Imbalanced dataset: ROC-AUC was used as the performance metric.

### Baseline Bag-of-Words model
- Logistic Regression model with regularization
- Feature importance
  - The following comments are more likely to be deleted:
    - Meaningless and repetitive comments: e.g., upvoted
    - Socially sensitive comments: e.g., rapes
    - Socially progressive comments: e.g., feminists and sjws
    - Politically sensitive comments: e.g., hitler
    - Comments involving swear words: e.g., fucking, bitches and hogwash
  - On the other hand, the comments that are less likely to be removed only contain auxiliaries.
  - Some Unicode characters are distorted when encoding.
    - Most tokenized features with positive coefficients contain the special character"Â", which represents the non-breaking spaces from the HTML template (the  s).
    - Some features with large negative coefficients, such as 0001f914, 0001f602 and fe0f.
    
### Tuned BoW model
- Tuned tf-idf, characters, n-grams
- Feature importance 
  - The following comments are more likely to be deleted:
    - Meaningless and repetitive comments: e.g., upvote and upvoted
    - Socially sensitive comments: e.g., commit suicide
    - Socially progressive comments: e.g., feminists
    - Politically sensitive comments: e.g., liberals, hitler, and hillary
    - Racially sensitive comments (not in baseline model): e.g., blacks and neanderthal
    - Comments involving swear words such as fuck and ass
    - Comments with jpg attachments (not in baseline model)
  - On the other hand, the comments that are less likely to be removed contain
    - Auxiliaries
    - Neutral or positive words (not in baseline model): e.g., abstract, curious, insects and hobbies
  - Like baseline model, some Unicode characters are distorted when encoding.

### Derived Features
- Four features were derived to further improve the model:
  - Punctuation: counts of  '!', '?', '*'
  - Capitalization: counts of capital letters 
  - Length of document
  - Use of urls: yes/no
- Feature importance
  - The pattern for comment removal does not change too much from the best model in the previous part. The derived features do not show in the above plot, so that they are not so important for prediction. Indeed, the two models only vary a little in ROC-AUC score.

## Task 2: Word Vectors
- A pre-trained Word2Vec model was used instead of the BoW model.
- The model does NOT improve. This may due to the way we encode the dataset, which generates a lot of special characters along with the original word. Therefore, there might be mismatches with the vocabulary library, leading to a worse score.
- Nevertheless, we should note that theoretically, we shall expect better performance by Word2vec, as it captures semantics of words and synonymous words as skip-gram model. Using cosine distance, it preserves text similarities better than bag of words approach.

## Summary

| Model  | Preprocessing | Tuning | Regularization | ROC-AUC | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Baseline: Logistic Regression BoW | `TfidVectorizer(stop_words='english', min_df=4, max_features=50000, use_idf=False)` | No | `C = 0.1` | 0.753314367159694 |
| Tuned BoW with `ngram_range=(1, 2)` and `sublinear_tf=True` | `TfidVectorizer(stop_words='english', min_df=4, max_features=50000, use_idf=False)` | No | Yes: `C = 0.1` | 0.753314367159694 | `TfidVectorizer(stop_words='english', min_df=4, max_features=50000, use_idf=False)` | Yes: `C`, `tf_idf`, `characters`, `n_grams` | Yes: `C = 1` | 0.7669171468063567 | 
| **Tuned BoW with derived features: punctuation, capitalization and use of urls** | `TfidVectorizer(stop_words='english', min_df=4, max_features=50000, ngram_range=(1, 2), use_idf=False, sublinear_tf=True)` | Yes: `C` | `C = 1` | **0.7678179485333205** |
| Pre-trained Word2Vec using Google News | None | Yes: `C` | `C = 1` | 0.7281880909523222 |

### Feature Importance by The Best Model
<img src="https://github.com/lullaby1024/Reddit_Comment_Removal/blob/master/img/output_43_0.png" width="90%">
