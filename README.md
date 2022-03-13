# Internship-at-polynomial-Drive-2022

#### try here:
- [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/bibhabasumohapatra/polynomial_app/main/app_lgbm.py)
## Step-1 Reading and pre-processing of the dataset 
Most difficult thing for me was to start at first, I already had a plan to use LSTM networks and wrote Dataset but everything came to Data at last. I was given a huge dataset massive 5 GB around thats also in json file, and lost hope to start with notebook,
But eventually now I had a solution, to not only convert but also compress. I went to kaggle created a notebook uploaded the json(took time) then ran import json import csv
and in the same file droped unnecessary columns, clicked save run and now I am using the csv file only of 2 GB then to 1.7 GB
https://www.kaggle.com/bibhabasumohapatra/amazon-dataset-csv-generator/data?select=PolynomialInternshipDrive2022.csv now, 
My efforts can be seen in my kaggle Notebook.
I have created a notebook in Kaggle named  _Amazon Dataset CSV generator_ now its public and the best thing is I ran the code with optimizing changes again and again to get best out of my Next comming model training. *Then I Ran and Saved the The Notebook* and clicked on new notebook and added data from selecting *Add data from your Notebooks' output file*
https://www.kaggle.com/bibhabasumohapatra/amazon-dataset-csv-generator/notebook

- I added another approach to my arsenal and THAT IS UNDERSAMPLING as data is too huge I am also trying to get equal number of outputs for every class for that I am dividing it into 5 csv files each of 1,2,3,4,and 5.   1->1161992 2->558228 3->742666 4->1297163 5 ->4927978
(if it works then I will update my models and weights)
## Step-2 Creating a classifier for the classification of Reviews into positive, negative, and neutral:

I tried the ways I know . . . Lightgbm, XGBOOST, Catboost, and tested with naive_bayes.  it worked great and lightgbm gave accuracy of 72 percent and what I did was simply use tf-idf and saved tf-idf and models in pkl and some in .bin files using joblib.
I am very well equipped with regression models and I use advanced ensemble techniques but I kept deployement in mind, ensembling different folds and different models and stacking and boosting them  can be done but then seeding should have to be done and I had to deploy 10-20 models simultaneusly and have to work more on post-process,  ensembling with classification data is my field of play though I use a special method not for calculating probaities but to do average and weighted averge of different models not using softmax rather I use sklearn's predict_proba (my excelent work can be checked here https://www.kaggle.com/bibhabasumohapatra/multiclass-using-predict-proba-ensemble-part-2) again this was too difficult in case of deployement to manage such 10 to pickled files, add also doing a stratified Kfold cross validation for such huge dataset was not a good Idea,  rather I follow of using parts with equal number of or balanced number of positive and negetive reviews.

as i said earlier I tried everything I knew , and one of the things I know well is PyTorch, I know PyTorch and worked on PyTorch, to mention how much I know PyTorch is - I know PyTorch DDP (distributed) and can even write custom collate_fn for dataloaders. But in case of bert models too I didn't leave it to that, rather I tried to run BERT and LSTM which was nearly impossible for a data this large to do it one day(not to mention ,need a day to run) but I did that and maybe I have got still too learn from Huggingface Transformer.(the code of BERT I tried was learnt and inspired from Abhishek Thakur's book Approaching (Almost) Any Machine Learning Problem.)

though I tried with chunks and then as much as I can hold to mention my catboost and lightgbm with only NLTK and Tfidfvectorizer(which I understand) worked better than expected around 72 with first try was not best, given BERT may get this situations in one hand and hugging face pipelines may not require fine-tuning(Just joking! and I wont dare too show my bert try)
- #### My attempts in this 24 hours journey is all in the ipynb-checkpoints folder in this repo itself.

- How I managed the stars - 1,2,3,4,5  and problem statement asking for positive, neutral and negetive be inferred
  - In my opinion why to waste data if some thing greater is to be achived which is more information about the data with not a difference of huge cost.
  - yes, it may tamper accuracy to 5-10 percentage, but I had to choose better accuracy versus better information that is if you process the 'overall' named column to 3 labels like positive(overall == 5 and 4), neutral(overall == 3) and negative(overall == 1 and 2) , in my opinion by doing this we camouflage the real datas existence for the reviewer.
  - what I did was
    - took data as it is. 5 classes 1,2,3,4 and 5
    - classified it as 5 classes
    - and to acknowledge the requirement of positive, negetive and neutral review, I used 5 classes as best as I can. I didn't simply write positive(4,5), neutral(3) and negetive(1,2)
    - I placed 1 -> 'negetive' ,2 -> 'negetive(marginally negetive)', 3 -> 'neutral', 4 -> 'positive' and 5 -> 'positive(very positive)'
   

## Step-3 Create a Confusion matrix and support training and Testing metrics:
- metric used was 
   - sklearn.metrics.accuracy_score
   - sklearn.metrics.confusion_matrix
- loss function torch.nn. nn.CrossEntropyLoss() but was not used in the deployement but it was worth learning or I say backpropagating.
#### Images
- LightbgmClf try 1 ==> Accuracy = 0.711375
![image](https://user-images.githubusercontent.com/68384968/154433253-dde76363-c653-4957-b0f2-8d154c506719.png)

- LightbgmClf try 2 Accuracy = 0.6709
![image](https://user-images.githubusercontent.com/68384968/154436053-222d9048-36a7-472a-af3d-ffa5d0ea61bd.png)

- CatboostCLf try 1  Accuracy = 0.6687
![image](https://user-images.githubusercontent.com/68384968/154466568-d115de9e-65b7-4241-88f0-e8e9aa2427da.png)

## Step-4 Deployment of the Model using some framework (Eg: Flask/ Django) on Heroku or any other service: 10 points as Bonus
## https://share.streamlit.io/bibhabasumohapatra/polynomial_app/main/app_lgbm.py
## [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/bibhabasumohapatra/polynomial_app/main/app_lgbm.py)
- I went with the other, these years I was too busy with PyTorch kaggle and Computer Vision never played with webpages as I did with writing neural networks and vision transformer with efficientNET
- ### Some sample Images 

![2](https://user-images.githubusercontent.com/68384968/154529850-464e5604-5b40-476b-aaa3-fe4b57b2efab.png)![1](https://user-images.githubusercontent.com/68384968/154530035-6d58a58a-4bd5-4849-92eb-2bd2c5295828.png)
- So what I used is Streamlit:  steps followed for inference were-
   - saving the the model trained not to forget vectorizers trained,
   - writing a input for python converting into dict then to pandas dataframe and try model.predict()
   - then printed negative for 1, negative(marginally negative) for 2, neutral for 3, and positive for 4 and positive(very positive) for 5
   - then ran a pipreqs command to run and it generated all the dependencies in requirement.txt
   - converted the input outputs with streamlit commands and saved the file.
   - now my folder has one inference py file with streamlit commands and requirement.txt and model pickle folder  and tfidf saved pickle
   - created a new repository https://github.com/bibhabasumohapatra/polynomial_app  with all files in above point
   - and deployed it on streamlit. *NOW ITS LIVE* works on mobile phone too.
   - https://share.streamlit.io/bibhabasumohapatra/polynomial_app/main/app_lgbm.py

