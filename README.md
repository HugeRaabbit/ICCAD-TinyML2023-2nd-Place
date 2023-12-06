### Training the Model
Our model training script is located in the Train_Model directory, specifically in the train_model_hlm.py file. 

To initiate the training process, you have to execute the train_model_hlm.py script with the following parameters: path_data and path_indices. These parameters specify the locations for the training dataset and indices respectively. Please ensure that these data files are placed in the correct directories before initiating the learning process.


### Deploying and Validating the Model on Target
We have made our own proprietary machine learning framework to validate the model on the target environment, without relying on the usage of X-Cube-AI.

For the model's implementation on the target, we have the framework present in the Model_on_target directory. The designed layout for this is given in the main.c file, which was created using the structural template extracted from TESTMODEL3.zip.

In our project architecture, the main.c file encompasses all the necessary functions required for providing a classification outcome. It’s important to note that there's no requirement to implement the Model_Init() method in this configuration. This differs from common practice, as our model doesn't have a neural network that needs to be loaded. Instead, all classifier coefficients are hardcoded directly into the main.c file in the extract_feature_peaks() function.

Our implementation focuses only on the application of the aiRun function for making inferences using the input IEMG segment. The rest of the code, which includes subprocesses such as data reception, data transmission, and serial communication, is kept consistent with the provided template. This strategic approach simplifies the deployment process, ensuring smooth functioning and enhanced performance when transferred to the target environment.

