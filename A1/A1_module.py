import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold

from skimage.feature import local_binary_pattern # Local Binary Pattern function

import cv2
import os

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.title(title)
    plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes)#, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    return plt

class A1:

    genders = ["male", "female"]

    def sort_test_data(self, labels_dir):
        # Create list of image names and corresponding gender classifications

        image_dic = pd.read_excel(labels_dir)
        print(image_dic)
        image_dic = image_dic[['img_name', 'gender']] # Choose columns which are of importance
        df = pd.DataFrame(image_dic)
        df.to_excel('./Datasets/test_source_genders_A1/test_labels_A1.xlsx',index=False)
        
        # Separate male and female images and corresponding labels into folders

        source_genders = pd.read_excel('./Datasets/test_source_genders_A1/test_labels_A1.xlsx')
        source_images_file_paths = glob.glob ("./Datasets/test_source_images_A1/*.jpg") #find all paths which match the given path
        source_images_file_paths = natsorted(source_images_file_paths) #sort the list of file names such that the image list will be in the correct order

        male_images = []
        female_images = []

        male_directory = "./Datasets/test_sorted_sets_A1/male/"
        female_directory = "./Datasets/test_sorted_sets_A1/female/"

        for file_path in source_images_file_paths:
            image = cv2.imread(file_path, cv2.COLOR_RGB2BGR) # read the image
            image_name = os.path.basename(file_path) # get the basename of the path file
            image_label = source_genders[source_genders['img_name']==image_name]['gender'].iloc[0]
            if(image_label == 1):
                male_images.append(image)
                directory = ''.join([male_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            else:
                female_images.append(image)
                directory = ''.join([female_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)

    def sort_train_data(self, labels_dir): 
        # Create list of image names and corresponding gender classifications
        image_dic = pd.read_excel(labels_dir)
        print(image_dic)
        image_dic = image_dic[['img_name.jpg', 'gender']] # Choose columns which are of importance
        df = pd.DataFrame(image_dic)
        df.to_excel('./Datasets/source_genders_A1/labels_A1.xlsx',index=False)
        
        # Separate male and female images and corresponding labels into folders

        source_genders = pd.read_excel('./Datasets/source_genders_A1/labels_A1.xlsx')
        source_images_file_paths = glob.glob ("./Datasets/source_images_A1/*.jpg") #find all paths which match the given path
        source_images_file_paths = natsorted(source_images_file_paths) #sort the list of file names such that the image list will be in the correct order

        male_images = []
        female_images = []

        male_directory = "./Datasets/sorted_sets_A1/male/"
        female_directory = "./Datasets/sorted_sets_A1/female/"

        for file_path in source_images_file_paths:
            image = cv2.imread(file_path, cv2.COLOR_RGB2BGR) #read the image
            image_name = os.path.basename(file_path)
            image_label = source_genders[source_genders['img_name.jpg']==image_name]['gender'].iloc[0]
            if(image_label == 1):
                male_images.append(image)
                directory = ''.join([male_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            else:
                female_images.append(image)
                directory = ''.join([female_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)

                
    def DNN_Face_Detection(self, file1,file2,gender):
            # write face detector NN to variable
            net = cv2.dnn.readNetFromCaffe("./deep-learning-face-detection/deploy.prototxt.txt",
                                    "./deep-learning-face-detection/res10_300x300_ssd_iter_140000.caffemodel")
            files = glob.glob("./Datasets/%s/%s/*.jpg" %(file1, gender)) #Get list of all images with gender
            filenumber = 0
            for f in files:
                # load the input image and construct an input blob for the image
                # by resizing to a fixed 300x300 pixels and then normalizing it
                image = cv2.imread(f)
                (h, w) = image.shape[:2]

                BGR_ave = image.mean(axis=(0,1)) # determine mean BGR values for mean subtraction
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (BGR_ave[2], BGR_ave[1], BGR_ave[0]))
                net.setInput(blob)
                detections = net.forward() # use net to detect face in images

                actual_detections = detections[0][0]
                detection_accuracies = detections[0][0][:,2]

                index_values = np.argsort(detection_accuracies)
                index_values = index_values[::-1]

                sorted_detections = [actual_detections[i] for i in index_values]
                sorted_detections = np.asarray(sorted_detections) # ordered detections by confidence level

                confidence = sorted_detections[0, 2]

                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence > 0.9:
                    # compute the (x, y)-coordinates of the bounding box for the object
                    box = sorted_detections[0, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cropped = image[startY:endY, startX:endX] #Cut the frame to size
                    try:
                        out = cv2.resize(cropped, (300, 300)) #Resize face so all images have same size
                        cv2.imwrite("./Datasets/%s/%s/%s.jpg" %(file2, gender, filenumber), out) #Write image - don' need to worry about keeping track of labels associated because already sorted
                        filenumber += 1 #Increment image number
                    except:
                        print("No face detected")
                        pass #If error, pass file            
                
    def face_detection(self, file1,file2):
        genders = ["male", "female"] #Define genders
        for gender in genders:
            self.DNN_Face_Detection(file1,file2,gender)    
                    
    def make_sets(self, file_name, r = 3):
        image_inputs = []
        image_labels = []
        for gender in self.genders:
            image_files = self.get_files(file_name,gender) # get file paths for images
            #Append data and generate labels 0-1
            for item in image_files:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                features = self.get_features(gray,r)
                image_inputs.append(features) #append image array to training data list
                image_labels.append(self.genders.index(gender))
                
        training_data, prediction_data, training_labels, prediction_labels = train_test_split(image_inputs,image_labels)
        return training_data, training_labels, prediction_data, prediction_labels,image_inputs,image_labels

    def get_files(self, file_name,gender): #Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("./Datasets/%s/%s/*.jpg" %(file_name, gender))
        return files

    def get_features(self, image, r = 3):
        M = 25
        N = 25
        tiles = [image[x:x+M,y:y+N] for x in range(0,image.shape[0],M) for y in range(0,image.shape[1],N)] # break up image into tiles of M*N pixels
        radius = r    
        no_points = 8*radius  # Number of points to be considered as neighbours
        eps=1e-7    
        global_hist_norm = []
        for tile in tiles:
            lbp = local_binary_pattern(tile, no_points, radius, method='uniform') # Uniform LBP is used
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, no_points + 2), range=(0, no_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps) # normalise histogram
            hist_norm = hist
            global_hist_norm.append(hist_norm)
        s = np.asarray(global_hist_norm).shape
        global_hist_norm = np.reshape(global_hist_norm,(s[0]*s[1]))
        return global_hist_norm

    def svc_param_selection(self, X, y, nfolds):
        Cs = [0.001, 0.01, 0.1, 1, 10,100,1000]
        gammas = [0.0001, 0.001, 0.01, 0.1,1,10, 'scale']
        decision_function_shapes = ['OVO', 'OVA']
        param_grid = {'C': Cs, 'gamma' : gammas, 'decision_function_shape': decision_function_shapes}
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        mean_CV_score = grid_search.cv_results_['mean_test_score']
        return best_params, mean_CV_score

    def train_model(self, xTrain, yTrain, C = 1, Gamma = 0.1, dfs = 'OVO'):
        # Paramters obtained using parameter tuning and learning curve optimisation
        c= C
        gamma = Gamma
        method = dfs
        clf = SVC(kernel='rbf',C = c , gamma = gamma,decision_function_shape = method)
        print ("training classifier")
        print ("size of training set is:", len(yTrain), "images")
        clf.fit(xTrain, yTrain)
        return clf

    def learning_curve_plot(self, x, y, c, gamma, dfs):
        
        title = "Learning Curve"

        kf = KFold(n_splits=5)

        estimator = SVC(kernel='rbf', C = c, gamma = gamma, decision_function_shape = dfs)
        plt = plot_learning_curve(estimator, title, x, y, ylim=(0.5, 1.01),
                            cv=kf, n_jobs=4)
        plt.show()

    def run_classifier(self, xTest,yTest, clf):
        yPredict = clf.predict(xTest)
        return accuracy_score(yTest,yPredict)

    def pp_train_data(self, parameter_list):
        pass

    def pp_test_data(self, parameter_list):
        pass

    def train(self):
        file_name = "A1_dataset_DNN"
        training_data_SVM, training_labels_SVM, prediction_data_SVM, prediction_labels_SVM,_,_ = self.make_sets(file_name, 1)
        clf = self.train_model(training_data_SVM, training_labels_SVM)
        accuracy = self.run_classifier(prediction_data_SVM, prediction_labels_SVM, clf)
        return accuracy, clf

    def test(self, clf):
        file_name = "test_A1_dataset"
        _, _, _, _, test_image_inputs, test_image_labels = self.make_sets(file_name, 1)
        accuracy = self.run_classifier(test_image_inputs, test_image_labels, clf)
        return accuracy
