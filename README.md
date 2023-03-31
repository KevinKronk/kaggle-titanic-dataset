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

## Data Collection

In this example we already have all of the data and it's a small dataset that is stored in a CSV file. So all we have to do is get the path to that file and load it up. 

If we didn't have the data, then we would have to get it from the logs of the ship, recorded survivors, or other researchers working on the problem. These sources would have to be verified for their trustworthiness and accuracy. We would also have to consider that the way we are collecting the data helps solve our problem while minimizing environmental noise or bias (like racial discrimination). There may also be government clearances or privacy concerns. If the data came from a variety of sources, then they would need to be converted to the same format and combined with the features properly matched. If this were a problem where we could collect more data, then we would have to consider if and when we would have enough to solve our problem. Additionally, if there was a lot of data, then we would need to consider where it's being stored (perhaps a relational database) and how to access and retrieve it. 