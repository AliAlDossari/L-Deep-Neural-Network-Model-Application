## Dogs vs. Monuments Classification
The below is an application that uses the `deep_nn_model` function, in addition to other helper functions, developed in the `nn_toolkit`. The addressed task covers the binary classification of Dogs images, obtained from the famous [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) competition image data, and various monuments images downloaded from Google image search using the term “monuments”.
The general application flow is as per the following:
1.	Importing necessary libraries.
2.	Reading in the images.
3.	Pre-processing and set up of train / test sets.
4.	Random check up on the processed images to ensure expected results are achieved.
5.	Standardizing the data for better training and fitting performance.
6.	Setting up the experiment to test multiple models with different hyperparameters of:
    * a.	Network Architecture (number of layers, neurons per layer).
    * b.	Number of Iterations (epochs).
    * c.	Learning Rates (alphas).
    * d.	Activation functions are chosen to be ReLU for all hidden layers and Sigmoid for the output layer.
7.	The resulting models are examined and the top performing ones are trained over longer epochs to analyse their test accuracies.
8.  The best model, based on test then train accuracies, is chosen and fine-tuned with regularizations if needed.
9.	New images, not used in the train or test sets, are used to validate the tuned final model.

Each of the above steps will be further explained as applied below. With that said, let’s get started with the first step.

## 1. Importing necessary libraries
The only module imported is the nn_toolkit as it internally imports all other nessacery libraries such as numpy, matplotlib, pandas…etc.
```python
from nn_toolkit import * # Has all the fuctions needed to set up the images as train and test data sets, build the model, train, test, evaluate, and predict.
np.random.seed(seed = 321) # To set up the seed for reproduction proposes.
%load_ext autoreload
%autoreload 2
```
## 2. Reading in the images
The total size of the data set is 800 images, 400 of dogs and 380 of monuments. For more information on the functions used below, the nn_toolkit modul has
all the documentation for each of them.
```python
path = getcwd() # get the working directory path which containes each set of images in a seperate folder.
dogs_path = path + '\\400DogPics\\' # 295 dogs images used as class 1.
monuments_path = path + '\\400MonumentPics\\' # 500 monumets images used as class 0.
```
## 3.	Pre-processing and set up of train / test sets.
```python
# Reading in the images from the above paths and transforming them into arrays of the shapes displayed in the output below.
dogs_array, dogs_labels, dogs_rejected_pics = prepare_image_data(dogs_path, resize = 100, show_rejected_images = False) # doges are labeled as 1.
monuments_array, monuments_labels, monuments_rejected_pics = prepare_image_data(monuments_path, resize = 100, label_tag = 0, show_rejected_images = False)
```
> Pics Array shape: (400, 100, 100, 3)
Labels Array shape: (1, 400)
Pics Array shape: (380, 100, 100, 3)
Labels Array shape: (1, 380)

```python
# Merging, shuffling, and splitting the data into train and test sets. Below output shows the dimensions of the resulted sets where 20% of the data is left for testing.
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y = merge_shuffle_split(dogs_array, dogs_labels, monuments_array, monuments_labels, validation_split = 0.2, seed = 42)
```
> Output Shapes:
train_set_x_orig: (624, 100, 100, 3)
train_set_y: (1, 624)
test_set_x_orig: (156, 100, 100, 3)
test_set_y: (1, 156)

## 4. Random check up on the processed images to ensure expected results are achieved.
```python
# Checking 6 images and thier labels form the train set.
random_image_check(num_images = 6, set_x = train_set_x_orig, set_y = train_set_y)
```
![Pic1](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/1.PNG)

```python
# Checking 6 images and thier labels form the test set.
random_image_check(num_images = 6, set_x = test_set_x_orig, set_y = test_set_y)
```
![Pic2](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/2.PNG)

The check-up clears and we can see that both the labelling and the dimensions are correct (dimensions are 100 * 100 in hight and width).
## 5. Standardizing the data for better training and fitting performance.
There are many methods that can be used to standardize the data, however, in the case of images the division of each pixil by 255 (max pixel value) is sufficient.
```python
# Flattening and Standerdizing the train_set_x_orig and test_set_x to better fit the model. Method = devision by the max pixel value 255.
train_set_x = prepare_image_arrays(train_set_x_orig)
test_set_x = prepare_image_arrays(test_set_x_orig)
```
> Shape of Flatten and Standardized array: (30000, 624)
> Shape of Flatten and Standardized array: (30000, 156)
## 6.	Setting up the experiment to test multiple models with different hyperparameters of:
     a.	Network Architecture (number of layers, neurons per layer).
     b.	Number of Iterations (epochs).
     c.	Learning Rates (alphas).
     d.	Activation functions are chosen to be ReLU for all hidden layers and Sigmoid for the output layer.
Setting up the experiment as follows:
* 6 model structures ranging from 1 to 4 hidden layers with arbitrary number of neurons: [1],  [4, 1],  [5, 1],  [8, 1],  [16, 5, 1],  [30, 17, 8, 1]
* Each with two sessions, one is for 300 epochs and the other is for 600 to observe the behaviour of the cost and accuracy.
* In each session, the model is trained 11 times over 11 different alphas form 0.001 up to 0.009 that are linearly equally spaced.
* The experiment will result in 6 * 2 * 11 = 132 models. These models are analysed and fillterd to a subset models.
* The filltered models are trained for longer epochs and the top performing (highest test then train accuracies) is chosen.
* The chosen model is tuned by training over long epoch and refining the overfitting by regularizing with L2 and/or dropouts.
* The tuned model is further validated by testing over new 'unseen' images for a final check.
* If the model checks out, it is then set and ready for produciton.
```python
# Setting up the experiment, for details on the 'deep_nn_model_exp' function, refer to the nn_toolkit repo.
exp_model_list = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[1], [4, 1], [5, 1], [8, 1], [16, 5, 1], [30, 17, 8, 1]],
                                   epochs_range = (300, 600), epochs_sets = 2, alpha_range = (0.001, 0.009), alpha_sets = 11,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 250, show_plots = True)
```
## 7. The resulting models are examined and the top performing ones are trained over longer epochs to analyse their test accuracies.
### Tabularizing the resulted models for examination
```python
# Tabularizing the resulted models using the 'models_summary' function for better examination.
models_df, top_model_index = models_summary(exp_model_list) # using the models_summary function to transform the models list into a pandas data frame.
models_df.sort_values(['Test Accuracy', 'Train Accuracy'], ascending = False).head(10) # sorting the models by test accuracy then train accuracy and getting the top 10.
```
![Pic3](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/3.PNG)
We can see that the top performing model is at index 34. However, further analysis is needed to examine the effects of longer epochs over different alphas on the costs and accuracies.
This is needed since the variant of model 34, the one with the same structure and alpha but over 600 epochs, is not within the top 10 shown in the tabel above, which may indicate the instability of this model.
Robust models should generaly benefit from longer training times, though not always. In this application however, the task is to come up with such robust models as they more consistently 
display expected behaviours.
 
 ```python
pd.set_option('display.max_columns', None) # to show all the columns of the dataframes preduced.
pd.set_option('display.max_rows', 10) # to show only 10 rows of the dataframes preduced.
```
In order to continue the analysis, will add two more columns to the models dataframe, mean and statandard deviation as follows:
```python
# Calculating the mean and standard deviation of the cost for every model and adding it as a new column
models_df['Avg Cost'] = models_df['Costs'].apply(lambda cost_list: np.mean(cost_list))
models_df['Std Cost'] = models_df['Costs'].apply(lambda cost_list: np.std(cost_list))
models_df.head(1)
```
![Pic4](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/4.PNG)
### Creating 4 new columns in the pivot dataframe to analyse whither or not:
1. Test and train accuracies increase over increased number of epochs.
2. Average cost decrease over increased number of epochs.
3. Standard deviation decrease over increased number of epochs.

The above 3 criteria comprise a very loose proxy for model stability. However, I have found them to be sufficient in narrowing the search space in an effective manner.
```python
# Creating criteria 1 for test accuracy.
models_df_pivot1['test_acc_imp'] = models_df_pivot1['Test Accuracy'][600] - models_df_pivot1['Test Accuracy'][300] # the '_imp' suffix abbreviates 'improvement'.

# Creating criteria 1 for train accuracy.
models_df_pivot1['train_acc_imp'] = models_df_pivot1['Train Accuracy'][600] - models_df_pivot1['Train Accuracy'][300]

# Creating criteria 2.
models_df_pivot1['avg_cost_imp'] = models_df_pivot1['Avg Cost'][300] - models_df_pivot1['Avg Cost'][600]

# Creating criteria 3.
models_df_pivot1['std_imp'] = models_df_pivot1['Std Cost'][300] - models_df_pivot1['Std Cost'][600]
models_df_pivot1
```
![Pic5](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/5.PNG)
### Summarizing over the above criteria by filltering the pivot data frame as follows:
```python
models_df_pivot1[(models_df_pivot1['avg_cost_imp'] > 0) & (models_df_pivot1['std_imp'] > 0) & (models_df_pivot1['test_acc_imp'] > 0)].sort_values(by = ['test_acc_imp', 'train_acc_imp', 'avg_cost_imp', 'std_imp'], ascending = False)
```
![Pic6](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/6.PNG)
We can see that the intial top model, indexed 34 in the models_df, is no longer part of the fillterd models above. This is consistant with the first conclusion we had about the model being instable.
 
 ### Running an experiment for each of the above resulted models by training over 3000 epochs and observing test accuracy
 ```python
 exp_model_list1 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[4, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.009, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic7](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/7.PNG)
```python
exp_model_list2 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[30, 17, 8, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.0066, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic8](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/8.PNG)
```python
exp_model_list3 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.0010, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic9](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/9.PNG)
```python
exp_model_list4 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[8, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.0082, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic10](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/10.PNG)
```python
exp_model_list5 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[16, 5, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.0018, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic11](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/11.PNG)
```python
exp_model_list6 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[30, 17, 8, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.0074, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic12](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/12.PNG)
```python
exp_model_list7 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[16, 5, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.0026, 0.009), alpha_sets = 1,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic13](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/13.PNG)
### Combaining the the experiments resluts into one data frame to be analysed
Now that the 7 experiments are done, we'll combine the results and analyse them as follows:
```python
filltered_models_df = pd.DataFrame() # setting the data frame
filltered_models_df = pd.concat([models_summary(exp_model_list1)[0], models_summary(exp_model_list2)[0],
                                models_summary(exp_model_list3)[0], models_summary(exp_model_list4)[0],
                                models_summary(exp_model_list5)[0], models_summary(exp_model_list6)[0],
                                models_summary(exp_model_list7)[0]], ignore_index = True, sort = False) # concatenating the models
filltered_models_df.sort_values(['Test Accuracy', 'Train Accuracy'], ascending = False) # sorting the models by test accuracy then train accuracy.
```
![Pic16](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/16.PNG)
Based on these results, the best model is at index 6, a 3 hidden layer model, which is the only one to achieve above 90% test accuracy. Additionally, it achieves 100% accuracy on the train data with 3000 epochs, the case with two other models, indecating that this number of epochs is enough for this particular data. However, the 100% fit maight be a signe of overfitting, but looking at the test accuracy it suggests otherwise, and since this is a separate issue, it is better handled in the fine-tuning step should resolve it if any.
Model 6 will now be chosen for fine tuning to positionally improve the test accuracy even further and finally validate it on random unseen images for the last check-up.
 
 ## 8.  The best model, based on test then train accuracies, is chosen and fine-tuned with regularizations if needed
Now will train model 6 one last time over 3000 epochs, since it seems to suffice, without adding any regularization effect since based on the high test accuracy, the model does not seem to suffer from overfitting, at least to the extent that a regularization is needed. Therefore, the fine-tuning will mainly be on the learning rate parameter alpha.

Fine-tuning of alpha will be to try several ones that are extremly close to the one that produced model 6, 0.0026, such as 0.00261, 0.002605, and 0.002601.
```python
tuning_model6 = deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y,
                                   layer_structures = [[16, 5, 1]],
                                   epochs_range = (3000, 600), epochs_sets = 1, alpha_range = (0.002601, 0.002605), alpha_sets = 5,
                                   lambd = 0.0, dropout_layers = [], keep_prob = 1.0,
                                   print_cost = True, print_every = 1000, show_plots = True)
```
![Pic17](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/17.PNG)
![Pic18](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/18.PNG)
![Pic19](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/19.PNG)
### Analysing the tuned models
```python
tuning_model6_df = models_summary(tuning_model6)[0]
tuning_model6_df.sort_values(['Test Accuracy', 'Train Accuracy'], ascending = False) # sorting the models by test accuracy then train accuracy.
```
![Pic20](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/20.PNG)
From the above, it seems that we manged to improve on the untuned model 6 by changing alpha from 0.0026 to 0.002601 which resulted in almost 1% improvement on the test accuracy! It's worth a thought that this minuscule change in the learning rate had this relatively huge jump in the accuracy. As for the other models, all were sub-bar compared to the untuned model 6. Therefore, the final model chosen is model indexed 0 in the above data frame.
 
 ```python
 final_model = tuning_model6_df.iloc[0] # saving the chosen model in 'final_model' variable to be used in predictions.
final_model
```
> Model No.               2020-04-03 18:52:03.461863
> Model Structure                  (30000, 16, 5, 1)
> Training Minutes                    0:02:45.422230
> Number of Parameters                             6
> Train X Shape                         (30000, 624)
>                                    ...            
> Test Accuracy                              91.0256
> Dropout Masks                                   {}
> Regularization Lambd                             0
> Keep Prob.                                       1
> Dropout Layers                                  ()
> Name: 0, Length: 18, dtype: object
##### 9.	New images, not used in the train or test sets, are used to validate the tuned final model
Now on to the final step to shed off any doubt about the final model, validating on a set of unseen images and observing the predictions.
```python
sample_path = path + '\\Test\\Test\\' # the path of the validation images.

Yhat = deep_nn_model_predict(sample_path, resize = 100, model = final_model) # for more info on this function, refer to repo nn_toolkit.

print('Yhat:', Yhat)
```
> Pics Array shape: (7, 100, 100, 3)
Labels Array shape: (1, 7)
Shape of Flatten and Standardized array: (30000, 7)

Yhat: [[0 0 1 1 0 0 0]]

![Pic21](https://github.com/AliAlDossari/L-Deep-Neural-Network-Model-Applictation/blob/master/application_pics/21.PNG)
