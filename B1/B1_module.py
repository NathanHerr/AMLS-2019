import numpy as np
import pandas as pd
import glob
import cv2
import dlib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from imutils import face_utils

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

class B1:
    
    face_shapes = ["shape0", "shape1", "shape2","shape3","shape4"]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file

    def sort_test_data(self, labels_dir):
        # Create list of image names and corresponding gender classifications
        image_dic = pd.read_excel(labels_dir)
        image_dic = image_dic[['file_name', 'face_shape']] # Choose columns which are of importance
        df = pd.DataFrame(image_dic)
        df.to_excel('./Datasets/test_source_shape_B1/test_labels_B1.xlsx',index=False)

        source_eye_color = pd.read_excel('./Datasets/test_source_shape_B1/test_labels_B1.xlsx')
        source_images_file_paths = glob.glob ("./Datasets/test_source_images_B1/*.png") #find all paths which match the given path
        source_images_file_paths = natsorted(source_images_file_paths) #sort the list of file names such that the image list will be in the correct order

        shape0_directory = "./Datasets/test_sorted_sets_B1/shape0/"
        shape1_directory = "./Datasets/test_sorted_sets_B1/shape1/"
        shape2_directory = "./Datasets/test_sorted_sets_B1/shape2/"
        shape3_directory = "./Datasets/test_sorted_sets_B1/shape3/"
        shape4_directory = "./Datasets/test_sorted_sets_B1/shape4/"

        for file_path in source_images_file_paths:
            image = cv2.imread(file_path, cv2.COLOR_RGB2BGR) #read the image
            image_name = os.path.basename(file_path)
            image_label = source_eye_color[source_eye_color['file_name']==image_name]['face_shape'].iloc[0]
            if(image_label == 0):
                directory = ''.join([shape0_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 1):
                directory = ''.join([shape1_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 2):
                directory = ''.join([shape2_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 3):
                directory = ''.join([shape3_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 4):
                directory = ''.join([shape4_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            else:
                print("no label")

    def sort_train_data(self, labels_dir):
        # Create list of image names and corresponding gender classifications
        image_dic = pd.read_excel(labels_dir)
        image_dic = image_dic[['file_name', 'face_shape']] # Choose columns which are of importance
        df = pd.DataFrame(image_dic)
        df.to_excel('./Datasets/source_shape_B1/labels_B1.xlsx',index=False)

        source_eye_color = pd.read_excel('./Datasets/source_shape_B1/labels_B1.xlsx')
        source_images_file_paths = glob.glob ("./Datasets/source_images_B1/*.png") #find all paths which match the given path
        source_images_file_paths = natsorted(source_images_file_paths) #sort the list of file names such that the image list will be in the correct order

        shape0_directory = "./Datasets/sorted_sets_B1/shape0/"
        shape1_directory = "./Datasets/sorted_sets_B1/shape1/"
        shape2_directory = "./Datasets/sorted_sets_B1/shape2/"
        shape3_directory = "./Datasets/sorted_sets_B1/shape3/"
        shape4_directory = "./Datasets/sorted_sets_B1/shape4/"

        for file_path in source_images_file_paths:
            image = cv2.imread(file_path, cv2.COLOR_RGB2BGR) #read the image
            image_name = os.path.basename(file_path)
            image_label = source_eye_color[source_eye_color['file_name']==image_name]['face_shape'].iloc[0]
            if(image_label == 0):
                directory = ''.join([shape0_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 1):
                directory = ''.join([shape1_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 2):
                directory = ''.join([shape2_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 3):
                directory = ''.join([shape3_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            elif(image_label == 4):
                directory = ''.join([shape4_directory,os.path.basename(image_name)])
                cv2.imwrite(directory, image)
            else:
                print("no label")
                
    def get_files(self, file_name, shape): #Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("./Datasets/%s/%s/*.png" %(file_name, shape))
        return files

    def euclid_distance(self, p1,p2):
        return distance.euclidean(p1, p2)

    def midpoint(self, p1, p2):
        return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]

    def get_landmarks(self, image):
        detections = self.detector(image, 1)
        landmarks_vectorised = []
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = self.predictor(image, d) #Draw Facial Landmarks with the predictor class
            shape = face_utils.shape_to_np(shape)
            
            forehead = self.euclid_distance(shape[19], shape[24])
            cheekbones = self.euclid_distance(shape[36], shape[45])
            jawline = self.euclid_distance(shape[0], shape[16])
            midpoint_eyebrows = self.midpoint(shape[19], shape[24])
            face_lenght = self.euclid_distance(midpoint_eyebrows, shape[8])
                    
            items = list(face_utils.FACIAL_LANDMARKS_IDXS.items())
            name, (i,j) = items[7] #the 7th item in the FACIAL_LANDMARKS_IDXS is the jaw line        
            landmarks_vectorised.append(forehead)
            landmarks_vectorised.append(cheekbones)
            landmarks_vectorised.append(jawline)
            landmarks_vectorised.append(face_lenght)
            landmarks_vectorised = np.asarray(landmarks_vectorised)
            landmarks_vectorised = np.append(shape[i:j], landmarks_vectorised)
            
        if len(detections) != 1:
            return []
        return landmarks_vectorised

    def make_sets(self, file_name1, file_name2):
        input_images = []
        labels = []
        for shape in self.face_shapes:
            print(" working on %s" %shape)
            files = self.get_files(file_name1, shape)
            filenumber = 0
            #Append data to training and prediction list, and generate labels 0-1
            for item in files:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                landmarks_vectorised = self.get_landmarks(gray)
                if (landmarks_vectorised != []):
                    cv2.imwrite("./Datasets/%s/%s/%s.jpg" %(file_name2, shape, filenumber), image) #Write image - don't need to worry about keeping track of labels associated because already sorted
                    input_images.append(landmarks_vectorised) #append image array to training data list
                    labels.append(self.face_shapes.index(shape))
                    filenumber += 1 #Increment image number                
        training_data, prediction_data, training_labels, prediction_labels = train_test_split(input_images,labels)        
        return training_data, training_labels, prediction_data, prediction_labels, input_images, labels

    def outlier_removal(self, image_inputs,image_labels):
        
        image_inputs = np.asarray(image_inputs)
        image_labels = np.asarray(image_labels)
        image_inputs_rs = np.asarray(image_inputs) #image_inputs.reshape(image_inputs.shape[0], image_inputs.shape[1])#*image_inputs.shape[2])#*image_inputs.shape[3])
        
        # Normalise data for clustering
        scaler = StandardScaler() 
        input_features_scaled = scaler.fit_transform(image_inputs_rs)
        df_input_features_scaled = pd.DataFrame(input_features_scaled) 
       
        dataframe = pd.DataFrame(input_features_scaled)
        dataframe["label"] = image_labels

        shape0_df = dataframe[dataframe['label'] == 0]
        shape1_df = dataframe[dataframe['label'] == 1]
        shape2_df = dataframe[dataframe['label'] == 2]
        shape3_df = dataframe[dataframe['label'] == 3]
        shape4_df = dataframe[dataframe['label'] == 4]

        shape0_inputs = shape0_df.drop(['label'], axis=1).values
        shape0_labels = shape0_df.filter(['label'], axis=1).values

        shape1_inputs = shape1_df.drop(['label'], axis=1).values
        shape1_labels = shape1_df.filter(['label'], axis=1).values

        shape2_inputs = shape2_df.drop(['label'], axis=1).values
        shape2_labels = shape2_df.filter(['label'], axis=1).values

        shape3_inputs = shape3_df.drop(['label'], axis=1).values
        shape3_labels = shape3_df.filter(['label'], axis=1).values

        shape4_inputs = shape4_df.drop(['label'], axis=1).values
        shape4_labels = shape4_df.filter(['label'], axis=1).values
        
        neigh = NearestNeighbors(n_neighbors=50)

        shape0_nbrs = neigh.fit(shape0_inputs)
        shape0_distances, shape0_indices = shape0_nbrs.kneighbors(shape0_inputs)
        shape0_distances = shape0_distances[:,1:]
        shape0_distances_mean = np.mean(shape0_distances, axis=1)
        shape0_distances_mean = np.sort(shape0_distances_mean, axis=0)

        shape1_nbrs = neigh.fit(shape1_inputs)
        shape1_distances, shape1_indices = shape1_nbrs.kneighbors(shape1_inputs)
        shape1_distances = shape1_distances[:,1:]
        shape1_distances_mean = np.mean(shape1_distances, axis=1)
        shape1_distances_mean = np.sort(shape1_distances_mean, axis=0)

        shape2_nbrs = neigh.fit(shape2_inputs)
        shape2_distances, shape2_indices = shape2_nbrs.kneighbors(shape2_inputs)
        shape2_distances = shape2_distances[:,1:]
        shape2_distances_mean = np.mean(shape2_distances, axis=1)
        shape2_distances_mean = np.sort(shape2_distances_mean, axis=0)

        shape3_nbrs = neigh.fit(shape3_inputs)
        shape3_distances, shape3_indices = shape3_nbrs.kneighbors(shape3_inputs)
        shape3_distances = shape3_distances[:,1:]
        shape3_distances_mean = np.mean(shape3_distances, axis=1)
        shape3_distances_mean = np.sort(shape3_distances_mean, axis=0)

        shape4_nbrs = neigh.fit(shape4_inputs)
        shape4_distances, shape4_indices = shape4_nbrs.kneighbors(shape4_inputs)
        shape4_distances = shape4_distances[:,1:]
        shape4_distances_mean = np.mean(shape4_distances, axis=1)
        shape4_distances_mean = np.sort(shape4_distances_mean, axis=0)
        
        # DBSCAN for outlier detection and removal

        shape0_dbscan = DBSCAN(eps = 5, min_samples = 38*3).fit(shape0_inputs) 
        shape0_outlier_labels = shape0_dbscan.labels_

        shape1_dbscan = DBSCAN(eps = 5, min_samples = 38*3).fit(shape1_inputs) 
        shape1_outlier_labels = shape1_dbscan.labels_

        shape2_dbscan = DBSCAN(eps = 5, min_samples = 38*3).fit(shape2_inputs) 
        shape2_outlier_labels = shape2_dbscan.labels_

        shape3_dbscan = DBSCAN(eps = 5, min_samples = 38*3).fit(shape3_inputs) 
        shape3_outlier_labels = shape3_dbscan.labels_

        shape4_dbscan = DBSCAN(eps = 5, min_samples = 38*3).fit(shape4_inputs) 
        shape4_outlier_labels = shape4_dbscan.labels_

        shape0_non_outlier_idx = np.where(shape0_outlier_labels == 0)[0]
        shape1_non_outlier_idx = np.where(shape1_outlier_labels == 0)[0]
        shape2_non_outlier_idx = np.where(shape2_outlier_labels == 0)[0]
        shape3_non_outlier_idx = np.where(shape3_outlier_labels == 0)[0]
        shape4_non_outlier_idx = np.where(shape4_outlier_labels == 0)[0]

        new_shape0_inputs_dbscan = np.asarray([shape0_inputs[i] for i in shape0_non_outlier_idx])
        new_shape0_labels_dbscan = np.asarray([shape0_labels[i] for i in shape0_non_outlier_idx])

        new_shape1_inputs_dbscan = np.asarray([shape1_inputs[i] for i in shape1_non_outlier_idx])
        new_shape1_labels_dbscan = np.asarray([shape1_labels[i] for i in shape1_non_outlier_idx])

        new_shape2_inputs_dbscan = np.asarray([shape2_inputs[i] for i in shape2_non_outlier_idx])
        new_shape2_labels_dbscan = np.asarray([shape2_labels[i] for i in shape2_non_outlier_idx])

        new_shape3_inputs_dbscan = np.asarray([shape3_inputs[i] for i in shape3_non_outlier_idx])
        new_shape3_labels_dbscan = np.asarray([shape3_labels[i] for i in shape3_non_outlier_idx])

        new_shape4_inputs_dbscan = np.asarray([shape4_inputs[i] for i in shape4_non_outlier_idx])
        new_shape4_labels_dbscan = np.asarray([shape4_labels[i] for i in shape4_non_outlier_idx])

        new_image_inputs = np.concatenate((new_shape0_inputs_dbscan, new_shape1_inputs_dbscan, new_shape2_inputs_dbscan, new_shape3_inputs_dbscan, new_shape4_inputs_dbscan))
        new_image_labels = np.concatenate((new_shape0_labels_dbscan, new_shape1_labels_dbscan, new_shape2_labels_dbscan, new_shape3_labels_dbscan, new_shape4_labels_dbscan))

        print(new_image_inputs.shape)
        print(new_image_labels.shape)
        print("Shape of training and testing data")

        new_training_data, new_prediction_data, new_training_labels, new_prediction_labels = train_test_split(new_image_inputs,new_image_labels)
        print(new_training_data.shape)
        print(new_training_labels.shape)
        print(new_prediction_data.shape)
        print(new_prediction_labels.shape)
        
        return new_training_data, new_prediction_data, new_training_labels, new_prediction_labels, new_image_inputs, new_image_labels

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

    def learning_curve_plot(self, x, y, c, gamma, dfs):
        
        title = "Learning Curve"

        kf = KFold(n_splits=5)

        estimator = SVC(kernel='rbf', C = c, gamma = gamma, decision_function_shape = dfs)
        plt = plot_learning_curve(estimator, title, x, y, ylim=(0.5, 1.01),
                            cv=kf, n_jobs=4)
        plt.show()

    def train_model(self, xTrain, yTrain, C = 100, Gamma = 0.001, dfs = 'OVO'):
        # Paramters obtained using parameter tuning and learning curve optimisation
        c= C
        gamma = Gamma
        method = dfs
        clf = SVC(kernel='rbf',C = c , gamma = gamma,decision_function_shape = method)
        print ("training classifier")
        print ("size of training set is:", len(yTrain), "images")
        clf.fit(xTrain, yTrain)
        return clf

    def run_classifier(self, xTest,yTest, clf):
        yPredict = clf.predict(xTest)
        return accuracy_score(yTest,yPredict)

    def pp_train_data(self, parameter_list):
        pass

    def pp_test_data(self, parameter_list):
        pass

    def train(self):
        file_name1 = "sorted_sets_B1"
        file_name2 = "B1_dataset"
        _, _, _, _, input_images, labels = self.make_sets(file_name1, file_name2)
        new_training_data, new_prediction_data, new_training_labels, new_prediction_labels, _, _ = self.outlier_removal(input_images, labels)
        clf = self.train_model(new_training_data, new_training_labels)
        accuracy = self.run_classifier(new_prediction_data, new_prediction_labels, clf)
        return accuracy, clf

    def test(self, clf):
        file_name1 = "test_sorted_sets_B1"
        file_name2 = "test_B1_dataset"
        _, _, _, _, input_images, labels = self.make_sets(file_name1, file_name2)
        _, _, _, _, new_image_inputs, new_image_labels = self.outlier_removal(input_images, labels)
        accuracy = self.run_classifier(new_image_inputs, new_image_labels, clf)
        return accuracy
