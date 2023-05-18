# Libraries imports
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

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
    for i in hand_only_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(hand_only_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(hand_only_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(hand_only_variables[count])
      axes[count].set_title(''+hand_only_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'whole-arm model':
    for i in whole_arm_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(whole_arm_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(whole_arm_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(whole_arm_variables[count])
      axes[count].set_title(''+whole_arm_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'egocentric hand only model':
    for i in egocentric_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(egocentric_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(egocentric_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(egocentric_variables[count])
      axes[count].set_title(''+egocentric_variables[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'egocentric whole arm model':
    for i in egocentric_variables_whole:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(egocentric_variables_whole[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(egocentric_variables_whole[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(egocentric_variables_whole[count])
      axes[count].set_title(''+egocentric_variables_whole[count]+' vs Time for '+title+'')
      axes[count].legend(loc='center right')

      count = count + 1

  elif title == 'joint-angles model':
    for i in joint_angles_variables:
        
      # Plot X vs. time in the for subplot
      axes[count].plot(time, y_test[:,count], label='{} actual'.format(joint_angles_variables[count]))
      axes[count].plot(time, y_pred[:,count],  label='{} predicted'.format(joint_angles_variables[count]))
      axes[count].set_xlabel('Time (ms)')
      axes[count].set_ylabel(joint_angles_variables[count])
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
  
  #Loop from 0 till 99 including for random state
  for i in range(100):
    # Split the dataset into training and testing sets, with test being 20 percent
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Specify custom alpha values from 0.1 till 5 by the step of 0.1
    alphas = np.arange(0.1, 5.1, 0.1)

    # Define the RidgeCV model with cross-validation
    ridge_cv = RidgeCV(alphas=alphas, cv=KFold(n_splits=5, shuffle=True, random_state=i)) # To shuffle, then random state can be used
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

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
    # if i == 50:
    #   plotPredictedAndTest(y_test, y_pred, title)
    if title == 'hand-only model':
      r2_hand_only.append(r2)
    elif title == 'whole-arm model':
      r2_whole_arm.append(r2)
    elif title == 'egocentric hand only model':
      r2_egocentric_hand_only.append(r2)
    elif title == 'joint-angles model':
      r2_joint_angles.append(r2)
    elif title == 'egocentric whole arm model':
      r2_egocentric_whole_arm.append(r2)

   
# This function receives an array of hand cartessian coordinates and transforms it to egocentric hand only coordinates
def egocentric_coordinates_hand_only(hand_position):
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

# This function receives an array of whole arm(hand and elbow) cartessian coordinates and transforms it to egocentric whole arm coordinates
# Egocentric because with respect to the observer(shoulder) - polar coordinates.
def egocentric_coordinates_whole_arm(whole_arm):  
    # Array for storing the calculated egocentric coordinates
    array_of_whole_arm_position=[]

    # Loop to calculate coordinates for each hand position
    for hand_x,hand_y,hand_z, elbow_x,elbow_y,elbow_z in whole_arm:
      #Calculate radius for each hand position
      hand_r = np.sqrt(hand_x**2 + hand_y**2 + hand_z**2)
      #Calculate polar angle for each hand position
      hand_theta = np.arccos(hand_z / hand_r)
      #Calculate azimuthal angle for each hand position
      hand_phi = np.arctan2(hand_y, hand_x)
      #Calculate radius for each elbow position
      elbow_r = np.sqrt(elbow_x**2 + elbow_y**2 + elbow_z**2)
      #Calculate polar angle for each elbow position
      elbow_theta = np.arccos(elbow_z / elbow_r)
      #Calculate azimuthal angle for each elbow position
      elbow_phi = np.arctan2(elbow_y, elbow_x)

      # Appends calculated value to array
      array_of_whole_arm_position.append([hand_r,hand_theta,hand_phi,elbow_r,elbow_theta,elbow_phi])

    return array_of_whole_arm_position

#This function performs Wilcoxon signed-rank test to compare two models
def wilcoxon_test(model_1, model_2, model_1_title, model_2_title):
  statistic, p_value = wilcoxon(model_1, model_2)
  if p_value < 0.05:
      print("There is a significant difference between the models (p-value = {:.4f})".format(p_value))
      median_diff = round((np.median(model_1) - np.median(model_2)), 5)
      if median_diff > 0:
        print("{} model has a median R-squared value of {:.5f}, which is {:.5f} higher than {} model".format(model_1_title.capitalize(),np.median(model_1), median_diff,model_2_title))
      else:
         print("{} model has a median R-squared value of {:.5f}, which is {:.5f} higher than {} model".format(model_2_title.capitalize(),np.median(model_2), abs(median_diff),model_1_title))
  else:
      print("There is no significant difference between the {} and {} models. The p-value = {:.4f} , which is higher than 0.05.".format(model_1_title,model_2_title,p_value))

#This function shows models matrix row wise.
def showModelsMatrix():
  # Define the values for models comparisons row wise
  models_values = np.array([[0, 1, 0, 1, 1],   # Hand only
                            [0, 0, 0, 1, 1],   # Whole arm
                            [1, 1, 0, 1, 1],   # 
                            [0, 0, 0, 0, 1],   #  
                            [0, 0, 0, 0, 0]])  # 
  # Define the headers for the rows and columns of the table
  headers = ['Hand-Only', 'Whole arm', 'Hand-only Egocentric', 'Joint Angles','Whole arm Egocentric']

  # Create the plot
  fig, ax = plt.subplots()
  im = ax.imshow(models_values, cmap=mcolors.ListedColormap(['white']))

  # Add the row and column headers as ticks
  ax.set_xticks(np.arange(len(headers)))
  ax.set_yticks(np.arange(len(headers)))
  ax.set_xticklabels(headers)
  ax.set_yticklabels(headers)

  # Rotate the tick labels and set their alignment
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Loop over the model values and add it to the plot
  for i in range(len(headers)):
      for j in range(len(headers)):
          text = ax.text(j, i, models_values[i, j],
                        ha="center", va="center", color="black")

  ax.set_title("Models matrix")
  plt.show()

  # Calculate the sum of values for each model
  models_sums = np.sum(models_values, axis=1)

  # Sort the model sums in descending order
  sorted_indices = np.argsort(models_sums)[::-1]
  sorted_sums = models_sums[sorted_indices]

  # Print the sorted models in the descending order
  print("Models in descending order:")
  for i in range(len(sorted_sums)):
      print(str(i+1)+".) "+headers[sorted_indices[i]])

# Beggining of main code

firing_rates = dict['f_rates']
elbow_position = dict['elbow_pos'] 
hand_position = dict['hand_pos']
joint_angles_variables = []
for variable in dict['jointNames'].flatten():
   joint_angles_variables.append(variable[0])
joint_angles = dict['joint_angle']

hand_only = hand_position
whole_arm = np.hstack((hand_position,elbow_position)) # put values one to another row wise
egocentric_hand_only = egocentric_coordinates_hand_only(hand_position)
egocentric_whole_arm = egocentric_coordinates_whole_arm(whole_arm)

# Preprocessing data - glazer https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7470933/
# Z-score firing rates
z_scored_firing_rates = scipy.stats.zscore(firing_rates)

# Names of coordinate systems variables
hand_only_variables = ['hand X','hand Y','hand Z']
whole_arm_variables = ['hand X','hand Y','hand Z', 'elbow X','elbow Y','elbow Z']
egocentric_variables = ['r', 'theta', 'phi']
egocentric_variables_whole = ['hand r','hand theta','hand phi','elbow r','elbow theta','elbow phi']

# R2 arrays for each ridge regression model
r2_hand_only = []
r2_whole_arm = []
r2_egocentric_hand_only = []
r2_egocentric_whole_arm = []
r2_joint_angles = []

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

ridge_regression(z_scored_firing_rates, hand_position, 'hand-only model')
ridge_regression(z_scored_firing_rates, whole_arm, 'whole-arm model')
ridge_regression(z_scored_firing_rates, egocentric_hand_only, 'egocentric hand only model')
ridge_regression(z_scored_firing_rates, joint_angles, 'joint-angles model')
ridge_regression(z_scored_firing_rates, egocentric_whole_arm, 'egocentric whole arm model')

# Conclusions - wilcoxon signed-rank test

wilcoxon_test(r2_hand_only, r2_whole_arm, 'hand only', 'whole arm')
wilcoxon_test(r2_hand_only, r2_egocentric_hand_only, 'hand only', 'egocentric hand only')
wilcoxon_test(r2_hand_only, r2_joint_angles, 'hand only', 'joint angles')
wilcoxon_test(r2_hand_only, r2_egocentric_whole_arm, 'hand only', 'egocentric whole arm')

wilcoxon_test(r2_whole_arm, r2_egocentric_hand_only, 'whole arm', 'egocentric hand only')
wilcoxon_test(r2_whole_arm, r2_joint_angles, 'whole arm', 'joint angles')
wilcoxon_test(r2_whole_arm, r2_egocentric_whole_arm, 'whole arm', 'egocentric whole arm')

wilcoxon_test(r2_egocentric_hand_only, r2_joint_angles, 'egocentric hand only', 'joint angles')
wilcoxon_test(r2_egocentric_hand_only, r2_egocentric_whole_arm, 'egocentric hand only', 'egocentric whole arm')

wilcoxon_test(r2_joint_angles, r2_egocentric_whole_arm, 'joint angles', 'egocentric whole arm')

showModelsMatrix()


# print(firing_rates.shape)
print("min r2_hand_only: ",min(r2_hand_only))
print("max r2_hand_only: ",max(r2_hand_only))
print("min r2_whole_arm: ",min(r2_whole_arm))
print("max r2_whole_arm: ",max(r2_whole_arm))
print("min hand egocentric: ",min(r2_egocentric_hand_only))
print("max hand egocentric: ",max(r2_egocentric_hand_only))
print("min r2_egocentric_whole_arm: ",min(r2_egocentric_whole_arm))
print("max r2_egocentric_whole_arm: ",max(r2_egocentric_whole_arm))
print("min r2_joint_angles: ",min(r2_joint_angles))
print("max r2_joint_angles: ",max(r2_joint_angles))








