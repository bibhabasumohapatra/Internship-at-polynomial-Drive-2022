# Internship-at-polynomial-Drive-2022

## Step-1 Reading and pre-processing of the dataset 
Most difficult thing for me was to start at first, I already had a plan to use LSTM networks and wrote Dataset but everything came to Data at last. I was given a huge dataset massive 5 GB around thats also in json file, and lost hope to start with notebook,
But evantually now I had a solution, to not only convert but also compress. I went to kaggle created a notebook uploaded the json(took time) then ran import json import csv
and in the same file droped unnecessary columns, clicked save run and now I am using the csv file only of 2 GB then to 1.7 GB
https://www.kaggle.com/bibhabasumohapatra/amazon-dataset-csv-generator/data?select=PolynomialInternshipDrive2022.csv now, 
My efforts can be seen in my kaggle Notebook.
I have created a notebook in Kaggle named  _Amazon Dataset CSV generator_ now its public and the beest thing is I ran the code with optimizing changes again and again to get best out of my Next comming model training. *Then I Ran and Saved the The Notebook* and clicked on new notebook and added data from selecting *Add data from your Notebooks' output file*
https://www.kaggle.com/bibhabasumohapatra/amazon-dataset-csv-generator/notebook


## Step-2 Creating a classifier for the classification of Reviews into positive, negative, and neutral:

I tried the ways I know . . . Lightgbm, XGBOOST, Catboost, and tested with naive_bayes.  it worked great and lightgbm gave accuracy of 72 percent and what I did was simply use tf-idf and saved tf-idf and models in pkl and some in .bin files using joblib.
I am very well equipped with regression models and I use advanced ensemble techniques but I kept deployement in mind, ensembling different folds and different models and stacking and boosting them  can be done but then seeding should have to be done and I had to deploy 10-20 models simultaneusly and have to work more on post-process,  ensembling with classification data is my field of play though I use a special method not for calculating probaities but to do average and weighted averge of different models not using softmax rather I use sklearn's predict_proba (my excelent work can be checked here https://www.kaggle.com/bibhabasumohapatra/multiclass-using-predict-proba-ensemble-part-2) again this was too difficult in case of deployement 




#### Images
- LightbgmClf try 1 ==> Accuracy = 0.711375
![image](https://user-images.githubusercontent.com/68384968/154433253-dde76363-c653-4957-b0f2-8d154c506719.png)


- LightbgmClf try 2 Accuracy = 0.6709
![image](https://user-images.githubusercontent.com/68384968/154436053-222d9048-36a7-472a-af3d-ffa5d0ea61bd.png)

- CatboostCLf try 1  Accuracy = 0.6687
![image](https://user-images.githubusercontent.com/68384968/154466568-d115de9e-65b7-4241-88f0-e8e9aa2427da.png)
