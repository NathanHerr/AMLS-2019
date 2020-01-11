# README -- AMLS_assignment_kit_19-20

## Organisation of project and Role of Each Folder/File

-> Each Task is separted into a folder each. Each folder contains their respective code (as well as any required auxiliary files), whichis set out as follows:

	A1.py: This module,in folder A1, contains the nesessary code to = pre-process the data, using funtions sort_train_data() and face_detection()
									= extract the features from the data, using make_sets(), which in turn makes calls to get_features() 
									= tune the parameters for the SVM model using svc_param_selection()
									= plot the learning curve for the model using learning_curve_plot(), which makes a call to plot_learning_curve()
									= train and test the final model using train() and test()

	A2.py: This module,in folder A2, contains the nesessary code to = pre-process the data, using funtions sort_train_data() and face_detection()
									= extract the features from the data, using make_sets(), which in turn makes calls to get_features() 
									= tune the parameters for the SVM model using svc_param_selection()
									= plot the learning curve for the model using learning_curve_plot(), which makes a call to plot_learning_curve()
									= train and test the final model using train() and test()

	B1.py: This module,in folder B1, contains the nesessary code to = pre-process the data, using funtions sort_train_data()
									= extract the features from the data, using make_sets(), which in turn makes calls to get_features()
									= detect and remove outliers from the data using outlier_removal()
									= tune the parameters for the SVM model using svc_param_selection()
									= plot the learning curve for the model using learning_curve_plot(), which makes a call to plot_learning_curve()
									= train and test the final model using train() and test()

	B2.py: This module,in folder B2, contains the nesessary code to = pre-process the data, using funtions sort_train_data() and get_eyes()
									= extract the features from the data, using make_sets(), which in turn makes calls to get_features()
									= detect and remove outliers from the data using outlier_removal()
									= tune the parameters for the SVM model using svc_param_selection()
									= plot the learning curve for the model using learning_curve_plot(), which makes a call to plot_learning_curve()
									= train and test the final model using train() and test()
 
-> The Dataset folder contains the original datasets provided as well as the final datasets
   (after performing pre-processing etc.) which are used as the final input to the system.
-> Main.py is used to run each model for Task A1, A2, B1 and B2 and report on their respective 
   performance.
-> Folders 'OpenCV_FaceCascade' and 'deep-learning-face-detection' contain required files for
   for certain steps to works, such as face detection.
-> File 'shape_predictor_68_face_landmarks' is used for landmark detection.

## Main Packages Required For Running the Code

-> cvutils
-> glob2                         
-> matplotlib                    
-> natsort                       
-> numpy                         
-> opencv-contrib-python         
-> opencv-python                 
-> pandas                       
-> scikit-image
-> scikit-learn               
-> sklearn

** Note if a package is missing - use pip, conda or any other package manager to install it **                       