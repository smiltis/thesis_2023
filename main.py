# Libraries imports
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from statistics import mean 



# Load .mat file and remove unnecessary columns
dict = scipy.io.loadmat('./data/BMI_Data_Smiltis.mat')
dict.pop('elbow_vel')
dict.pop('hand_vel')
dict.pop('joint_vel')


def plot_3D_example(x,y,z, title):
  # Create the 3D scatterplot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Plot the scatterplot
  ax.scatter(x[0], y[0], z[0], c='r', marker='o')
  ax.scatter(x[1:], y[1:], z[1:], c='b', marker='o')

  # Connect the dots with lines
  for i in range(len(x)-1):
      ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], 'k--')

  # Set labels and title
  ax.set_xlabel('X Axis')
  ax.set_ylabel('Y Axis')
  ax.set_zlabel('Z Axis')
  ax.set_title(title)

  # Show the plot
  plt.show()

def plot_joints():
    # Generate some example data
  time = np.arange(0, 100*9.8, 9.8) 
  shoulder_adduction = joint_angles[:100,0] 
  shoulder_rotation = joint_angles[:100,1] 
  shoulder_flexion = joint_angles[:100,2] 
  elbow_flexion = joint_angles[:100,3] 
  radial_pronation = joint_angles[:100,4] 
  wrist_flexion = joint_angles[:100,5]
  wrist_abduction = joint_angles[:100,6]

  # Create the time series plot
  fig, ax = plt.subplots(figsize=(12, 6))

  # Plot the joint angles over time
  ax.plot(time, shoulder_adduction, label='Shoulder Adduction')
  ax.plot(time, shoulder_rotation, label='Shoulder Rotation')
  ax.plot(time, shoulder_flexion, label='Shoulder Flexion')
  ax.plot(time, elbow_flexion, label='Elbow Flexion')
  ax.plot(time, radial_pronation, label='Radial Pronation')
  ax.plot(time, wrist_flexion, label='Wrist Flexion')
  ax.plot(time, wrist_abduction, label='Wrist Abduction')

  # Set labels and title
  ax.set_xlabel('Time ms')
  ax.set_ylabel('Joint Angles degrees')
  ax.set_title('Joint Angles during 9.8s of whole arm movement')

  # Add a legend
  ax.legend()

  # Show the plot
  plt.show()

def ridge_regression(X, y, title):
  # casting lists to np arrays, because with list it gives an error of indexing
  X=np.array(X)
  y=np.array(y)
  alpha_values = np.linspace(0.1, 1, num=10) # alpha values from 0.1 till 3 by the step of 0.1

  best_r_avg_square = -1
  best_r_square_avg_alpha = -1

  for alpha in alpha_values:
    # Creates a Ridge regression model
    
    ridge = Ridge(alpha=alpha) # alpha=regularization strength - 1 default value - test on multiple to find the best fit

    # Set up k-fold cross-validation
    k = 4  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r_scores = []  # List to store R squared scores for each fold

    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # Model fitting to the training data
      ridge.fit(X_train, y_train)

      # Predict dependent variables based per model for test data
      y_pred = ridge.predict(X_test)
    
      # Calculate R-squared for dependent variables predictions and append it to the r_scores array
      r2 = r2_score(y_test, y_pred,  multioutput='variance_weighted')   # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
      r_scores.append(r2)

    # Calculate average R squared value across all K-folds
    avg_r2 = np.mean(r_scores)

    # print("R Squared score for each fold:", r_scores)
    print("Average R Squared score across all",title,"K-folds with alpha value of", alpha,":", avg_r2)
    if avg_r2 > best_r_avg_square:
       best_r_avg_square = avg_r2
       best_r_square_avg_alpha = alpha
  print("Best Average R Squared score across all",title,"K-folds with alpha value of", best_r_square_avg_alpha,":", best_r_avg_square)


# This function receives an array of hand cartessian coordinates and transforms it to egocentric coordinates
def egocentric_coordinates(hand_position):
    # Array for storing the calculated egocentric coordinates
    array_of_hand_position=[]

    # Loop to calculate coordinates for each hand position
    for x,y,z in hand_position:
      #Calculate radius for each hand position
      r = np.sqrt(x**2 + y**2 + z**2)
      #Calculate polar angle for each hand position
      theta = np.arccos(z / r)
      #Calculate azimuthal angle for each hand position
      phi = np.arctan2(y, x)
      # Appends calculated value to array
      array_of_hand_position.append([r,theta,phi])

    return array_of_hand_position



# Beggining of code

# print("The dictionary values: ", list(dict)[3:])
# print(dict['jointNames'])

firing_rates = dict['f_rates']
elbow_position = dict['elbow_pos'] 
hand_position = dict['hand_pos']
joint_names = dict['jointNames']
joint_angles = dict['joint_angle']

hand_only = hand_position
whole_arm = np.hstack((hand_position,elbow_position)) # put values one to another row wise
egocentric = egocentric_coordinates(hand_position)

# Preprocessing data - glazer https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7470933/
# Z-score firing rates
z_scored_firing_rates = scipy.stats.zscore(firing_rates)


# Plotting example data
# x = hand_position[:100,0].tolist()
# y = hand_position[:100,1].tolist()
# z = hand_position[:100,2].tolist()
# plot_3D_example(x,y,z,'3D Scatterplot of hand position during 9.8s of movement')

# x = elbow_position[:100,0].tolist()
# y = elbow_position[:100,1].tolist()
# z = elbow_position[:100,2].tolist()
# plot_3D_example(x,y,z,'3D Scatterplot of elbow position during 9.8s of movement')

# plot_joints()

# Ridge regression for different models pass data to parameters

ridge_regression(z_scored_firing_rates,hand_position, 'hand-only model')
ridge_regression(z_scored_firing_rates,whole_arm, 'whole-arm model')
ridge_regression(z_scored_firing_rates,egocentric, 'egocentric model')
ridge_regression(z_scored_firing_rates,joint_angles, 'joint-angles model')
# print()








