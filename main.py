# Libraries imports
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
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

def plotPredictedAndTest(y_test, y_pred, title):
  # Generate time data
  time = np.arange(0, y_test.shape[0]*9.8, 9.8) 

  # Create a figure with x subplots
  fig, axes = plt.subplots(y_test.shape[1], 1, figsize=(16, y_test.shape[1]*3))
  
  count = 0
  if title == 'hand-only model':
    for variable in hand_only_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(hand_only_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(hand_only_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(hand_only_variables[count])
      # axes[count].set_title('{} vs Time for '.format(hand_only_variables[count], title))
      axes[count].set_title(''+hand_only_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'whole-arm model':
    for variable in whole_arm_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(whole_arm_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(whole_arm_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(whole_arm_variables[count])
      # axes[count].set_title('{} vs Time for {}'.format(whole_arm_variables[count], title))
      axes[count].set_title(''+whole_arm_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'egocentric model':
    for variable in egocentric_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(egocentric_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(egocentric_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(egocentric_variables[count])
      # axes[count].set_title('{} vs Time for {}'.format(egocentric_variables[count], title))
      axes[count].set_title(''+egocentric_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'joint-angles model':
    for variable in joint_angles_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(joint_angles_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(joint_angles_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(joint_angles_variables[count])
      # axes[count].set_title(''+joint_angles_variables[count]+' vs Time for '+title+'')
      axes[count].set_title(''+joint_angles_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1
        
  # Adjust spacing between subplots
  plt.subplots_adjust(hspace=2)

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
  ax.legend(loc='center right')

  # Show the plot
  plt.show()

def ridge_regression(X, y, title):
    # casting lists to np arrays, because with list it gives an error of indexing
    X=np.array(X)
    y=np.array(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Specify custom alpha values
    alphas = np.arange(0.1, 5.1, 0.1)

    # Define the RidgeCV model with cross-validation
    ridge_cv = RidgeCV(alphas=alphas, cv=KFold(n_splits=5, shuffle=True, random_state=42))

    # Fit the RidgeCV model on the training set
    ridge_cv.fit(X_train, y_train)

    # Get the best alpha value from RidgeCV
    best_alpha = ridge_cv.alpha_

    # Train the Ridge model with the best alpha on the entire training set
    ridge_best = Ridge(alpha=best_alpha)
    ridge_best.fit(X_train, y_train)

    # Evaluate the Ridge model on the testing set
    y_pred = ridge_best.predict(X_test)
    r2 = r2_score(y_test, y_pred,  multioutput='variance_weighted')

    #Plot predicted and actual data
    plotPredictedAndTest(y_test, y_pred, title)
    print(y_test[0])
    print(y_pred[0])

    # Print the best alpha value and the R2 score on the testing set
    print("Best Average R Squared score across all",title,"K-folds with alpha value of", best_alpha,":", r2)



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

firing_rates = dict['f_rates']
elbow_position = dict['elbow_pos'] 
hand_position = dict['hand_pos']
joint_angles_variables = []
for variable in dict['jointNames'].flatten():
   joint_angles_variables.append(variable[0])
joint_angles = dict['joint_angle']

hand_only = hand_position
whole_arm = np.hstack((hand_position,elbow_position)) # put values one to another row wise
egocentric = egocentric_coordinates(hand_position)

# Preprocessing data - glazer https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7470933/
# Z-score firing rates
z_scored_firing_rates = scipy.stats.zscore(firing_rates)

# Names of coordinate systems variables
hand_only_variables = ['hand X','hand Y','hand Z']
whole_arm_variables = ['hand X','hand Y','hand Z', 'elbow X','elbow Y','elbow Z']
egocentric_variables = ['r', 'theta', 'phi']

# Plotting example data
x = hand_position[:100,0].tolist()
y = hand_position[:100,1].tolist()
z = hand_position[:100,2].tolist()
plot_3D_example(x,y,z,'3D Scatterplot of hand position during 9.8s of movement')

x = elbow_position[:100,0].tolist()
y = elbow_position[:100,1].tolist()
z = elbow_position[:100,2].tolist()
plot_3D_example(x,y,z,'3D Scatterplot of elbow position during 9.8s of movement')

plot_joints()

# Ridge regression for different models pass data to parameters

ridge_regression(z_scored_firing_rates,hand_position, 'hand-only model')
ridge_regression(z_scored_firing_rates,whole_arm, 'whole-arm model')
ridge_regression(z_scored_firing_rates,egocentric, 'egocentric model')
ridge_regression(z_scored_firing_rates,joint_angles, 'joint-angles model')










