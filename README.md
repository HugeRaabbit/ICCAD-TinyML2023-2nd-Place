### How to train our model？
The training script is present in Train_Model\train_model_hlm.py. To run the training, simply run the python script train_model_hlm.py with following parameters path_data and path_indices.

### How to validate model on target
We didn't use X-Cube-AI, we just use our own framework.

In the folder Model_on_target, we have the design implemented in the main.c and used the template provided by TESTMODEL3.zip. In the project, the file main.c contains all the functions needed for classification result.

Note: no need to implement Model_Init() method is the function as we do not have any neural network to be loaded, rather all our classifier coefficient are hardcoded in the main.c, extract_feature_peaks() function.
We only impelemnted aiRun function to inference the input IEMG segment. The rest of the code, including data reception, data transmission and serial communication, is retained as a template.





