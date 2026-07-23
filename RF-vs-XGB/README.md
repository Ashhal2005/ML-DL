What i learned? 

----XgBoost is faster than Random Forest---- #chatgpt expalined me why
1. Random Forest finds best split for each row making it more computational
   While XGBoost does Histogram Binning
2. Regularization in XGBoost prevents tree from growing deep
3. XGBoost does pruning
4. XGBoost is memory efficient