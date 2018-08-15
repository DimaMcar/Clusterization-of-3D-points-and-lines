from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import numpy as np
import os
from collections import Counter



#Open file to a list
def open_files(files):
    points = 0
    for fl in files:
        
        with open(fl) as f:
            
            for line in f:            
                test = line.rstrip("\n").split(",")
                data.append(test) 
                points+=1
                
    return points
                
#Check if string is a number       
def isNumber(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#Remove lines from the data of points    
def remove_lines():    
    rawLines = []
    for point in data:
        for x in point:
            if not isNumber(x):
                rawLines.append(point)
                break
    for point in rawLines:
        data.remove(point)
        
    return rawLines    



#transform raw list to the list of 4 points
# (the last 2 are beginning and ending of line)
def rawToLines(raw):
    for point in raw:
        temp = point[2].split("-")
        point.remove(point[2])
        for num in temp:
            if isNumber(num):
                point.append(num)
    return raw

def reject_outliers(lines):
    temp = lines
    ind = 0
    for index, line in enumerate(lines, start=0):
        if line[2]>200 or line[3]>200:
            temp = np.delete(temp, index-ind, axis = 0)
            
            ind+=1
            
    return temp

      
#plot points after opening to list in 3d
def plot_initial(lines):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[:,0], data[:,1], data[:,2], 'o', markerfacecolor='r',
                 markeredgecolor='k', markersize=1)
    for line in lines:        
        ax.plot([line[0],line[0]], [line[1],line[1]], [line[2],line[3]], c="b")
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.title("Initial represantation after loading dataset")
    
    return


#predict clusters using Bayesian Gausian Mixture
def predictBayesian(data2D):    
    bgmm = mixture.BayesianGaussianMixture(n_components=7, covariance_type='diag',
                                        weight_concentration_prior = 1e-5, 
                                        max_iter=10000, random_state=1).fit(data2D)

    labels=bgmm.predict(data2D)
    return bgmm, labels



#plot correct prediction
def plot_correct (unique_labels):

    colors = [plt.cm.nipy_spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    
        
    for k, col in zip(unique_labels, colors):
    
        class_member_mask = (labels == k)
    
        xyz = data[class_member_mask] #& core_samples_mask
        plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=4)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title("Estimated number of clusters: 6")
    


#Subfunction for ploting weights
def plot_results(ax2, estimator, title):

    ax2.set_title(title)
    ax2.get_xaxis().set_tick_params(direction='out')
    ax2.yaxis.grid(True, alpha=0.7)
    for k, w in enumerate(estimator.weights_):
        ax2.bar(k, w, width=0.9, color='#56B4E9', zorder=3,
                align='center', edgecolor='black')
        ax2.text(k, w + 0.007, "%.1f%%" % (w * 100.),
                 horizontalalignment='center')
    
    ax2.set_ylim(0., 1.1)
    ax2.tick_params(axis='y', which='both', left='off',
                    right='off', labelleft='off')
    ax2.tick_params(axis='x', which='both', top='off')

    
    ax2.set_ylabel('Weight of each component')

#Plot weights to see why the clustering selects 6 clusters
def plot_weights(weights):
    plt.figure(3, figsize=(4.7 * 3, 8)) 
          
    plt.subplots_adjust(bottom=.04, top=0.90, hspace=.05, wspace=.05,
                        left=.03, right=.99)
    
    
    gs = gridspec.GridSpec(3, len(weights))    
    
    for k, concentration in enumerate(weights):
        bgmm.weight_concentration_prior = concentration
        bgmm.fit(data2D)
        plot_results(plt.subplot(gs[2,k]), bgmm, "gamma = "+str(concentration)) 
        
    

#transform lines to points with interval 0.01 and get prediction for each point
def estimate_lines():
    pointsLines = []    
    lines2D = np.vstack((lines[:,1], lines[:,2], lines[:,3])).T
    for line in lines2D:
        i = 0.01
        
        if line[1]<=line[2]:
            while i <= line[2]:
                pointsLines.append([line[0], line[1]+i])
                i+=0.01
        if line[1] > line[2]:
            while i > line[2]:
                pointsLines.append([line[0], line[1]-i])
                i+=0.01
    pointsLines = np.asarray(pointsLines)
    linesLabels = bgmm.predict(pointsLines)
    
    return pointsLines, linesLabels


 
os.chdir(".\\Demo")
files = glob.glob("*")

data = []  

#Let's load our data into list
points = open_files(files)
print("Total number of points is: "+str(points))
rawLines = remove_lines()

data = np.asarray(data)
data = np.asarray(data, dtype="int") 
print("Number of precise points is "+str(data.shape[0]))
print("Number of unprecise points (lines) is "+str(len(rawLines)))


lines = rawToLines(rawLines)
lines = np.asarray(lines, dtype="int")

#Let's visualize our data of precise points
#First we need to reject outliers
draw_lines = reject_outliers(lines)
plot_initial(draw_lines)
#We can visually see 6 clusters as planes in parallel to xy plane
#However we still need to prove there are really 6 clusters
#Let's project dataset to yz plane
data2D = np.vstack((data[:,1],data[:,2])).T

#Since we need dynamic number of clusters, 
#the distance between clusters isn't very large,
#and clusters aren't uniformly shaped, Bayesian Gausian Mixture is a reasonable choice
bgmm, labels = predictBayesian(data2D)
unique_labels = set(labels)
print("These are unique labels: "+str(unique_labels))
print("These are possible clusters for precise points: - "+str(Counter(labels)))

#Let's see the plotting
plot_correct (unique_labels)

#Plotting look good, but let's check the weights that are dynamicaly adapted
weights = [1e-8, 1e-6, 1]
plot_weights(weights)
#We see that although we have chosen 7 clusters(components), 
#it's dyanamically eliminated by weights being zero or close to zero

#Let's deal with unprecise points. Now we can predict most probable cluster
# for each point in line (unprecise points)

#Let's divide lines into points and predict possible clusters for them
pointsLines, linesLabels = estimate_lines()
print("These are possible clusters for lines(unprecise points) is" +str(Counter(linesLabels)))
#It's clear that 2 clusters correspond to initial visual representation of lines
# that intersected or was near two planes


plt.show()      
