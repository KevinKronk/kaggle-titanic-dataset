# kaggle-titanic-dataset

Titanic - Machine Learning from Disaster - Kaggle Competiton 

https://www.kaggle.com/c/titanic/overview

I run through the entire machine learning project structure on the Titanic dataset in the Jupyter Notebook. 

![image](https://user-images.githubusercontent.com/41022783/117377329-057d5680-aea1-11eb-908e-905342c0073e.png)

![image](https://user-images.githubusercontent.com/41022783/117377371-1c23ad80-aea1-11eb-95cd-d760994be449.png)

## The Big Picture and Framing the Problem

This dataset comes from the Titanic Kaggle competition. The goal is relatively simple: Build a predictive model that is able to predict which passengers survived the Titanic disaster based on their passenger data. 

While this is a simple competition dataset, it does have potential real world applications. This type of data may be collected in a study proposed by the government to learn more about a disaster. Alternatively, it could be a business trying to learn from a failure, or simply learning more about their data. Either way, the objective would be a deeper understanding of the data and what that means in context. For a disaster, this might serve as a guiding force for future disaster response in terms of resource allocation, expectations, and considerations. For a business, the problem could have been about product defects on an assembly line based on processing steps. In that case, the model could be used to make adjustments that help eliminate defects, reduce costs, and make future predictions. 

The goal of this specific project is to accurately predict which passengers survived, with an understanding of how the model made that decision. Because the Titanic disaster is a closed off event, it's unlikely the model trained could be used on similar future data (like a modern cruise ship sinking). This is a problem in which we already have all of the data and just want to make predictions on the test set labels. Since the data comes with labels, this is a supervised classification problem. The accuracy of the predictions on the test set will serve to measure performance. 