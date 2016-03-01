# XGBoost Model Tuning Template  
A template code for tuning the XGBoost Model. The code is based on a competition on Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge  

After tuning three parameters of the XGBoost model, I got a not bad score:  
![alt tag](https://github.com/SauceCat/XGBoost-Model-Tuning-Template/blob/master/xgboost_score.png)

Certainly, the model itself has much more parameters, which are all potentially to be tuned. However, I think the most importance parameters are:  

**eta [default=0.3]**  
step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.
range: [0,1]  

**max_depth [default=6]**  
maximum depth of a tree
range: [1,∞]  

**subsample [default=1]**  
subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
range: (0,1]  

For further information about the parameters, you can refer to [XGBoost Parameters](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md).
