To run the pipeline please follow these step:

1) Install the TensorFlow. Follow the instruction from here: https://www.tensorflow.org/install/

2) Pre-trained Models of CNN architectures should be downloaded from the "Pre-trained Models" part of https://github.com/wenwei202/terngrad/tree/master/slim#pre-trained-models and be located in your machine (e.g. GitHub_CNN_Smoothie/scripts/slim/run/checkpoint). The files for pre-trained models are available under the column named "Checkpoint".

3) Divide the images with the original size into two or more classes based on the aim of classification (e.g., discrimination of adenocarcinoma and squamous cell lung cancer). Some images in each class should be selected as Train (train and validation) and Test sets. Therefore, the name of images as well as their path should be save as .txt file.

4) _NUM_CLASSES should be set in tumors.py (this script is located in CNN_Smoothie/scripts/slim/datasets).

5) Run the convert.py (it is located in the "CNN_Smoothie/scripts" directory) to allocate the suitable percentage of images to train and validation sets.the convert.py needs three arguments including: the address of images for training, the address of where the result will be located, and the percentage of validation images for the training step (e.g., $ python convert.py ../Images/train process/ 30). It will save converted .tf records in the "process" directory.

6) The CNN algorithm (e.g., Inception architecture) should be run on the Train set images from the "CNN_Smoothie/scripts/slim" directory. First got the the following directory: CNN_Smoothie/scripts/slim. Then open load_inception_v1.sh located in "run/" directory and edit PRETRAINED_CHECKPOINT_DIR,TRAIN_DIR, and DATASET_DIR addresses. See the load_inception_v1.sh, for instance. Then, run the following command in shell script: 
$ ./run/load_inception_v1.sh

* If you got the bash error like permission denied, run the following line in your shell:
$ chmod 777 load_inception_v1.sh

* Each script in slim dataset should be run separately based on the selected architecture. The slim folder contains some sub-folders. 

* You can set up the parameters of each architectures in “run” sub-folder. For example you can set the architecture in a way to run from scratch or trained for the last or all layer. Also you can set the batch size or the number of maximum steps. 

* see the result folder at scripts/result as the result of running the above script.

* Note that the flag for --clone_on_cpu is set to "True". If you are going to use GPUs you should change this flag to "False".

7) The trained algorithms should be tested using test set images. In folder "CNN_Smoothie/scripts/slim", predict.py loads a trained model on provided images. This code get 5 arguments:
$ python predict.py v1 ../result/ ../../Images/test output.txt 2

* v1 = inception-v1, ../Images/test = the address of test set images, out.txt = the output result file, 2 = number o classes

* You can see output.txt in "GitHub_CNN_Smoothie/scripts/slim", for example.

* If you are going to run predict.py on CPU add os.environ['CUDA_VISIBLE_DEVICES'] = '' before session = tf.Session() in the predict.py.

8) The accuracy can be measured using accuracy measurement codes.

9) The plot can be illustrated using plot illustration codes.





