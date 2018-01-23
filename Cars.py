import numpy as np
import pandas as pd
 
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

dfcar = pd.read_csv('cardata1.csv',sep=',', header=0 )
dfcar.info()
dfcar.head()




# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# As you can see from the summary, all are string objects, we want them to be converted to numeric unqiue values. In pandas you can use factorize() function to encode the labels columnwise or a simple replace() will work.
# 
# Lets encode the labels in the dataset like this, in a very simple way…

# vhigh = 4  high=3  med=2  low=1
# 
# 5more = 6    more =5
# 
# small =1   med=2   big=3
# 
# unacc=1   acc=2  good=3   vgood=4

# # Summarize the Dataset
# 
# Dimensions of the dataset.
# 
# Peek at the data itself.
# 
# Statistical summary of all attributes.
# 
# Breakdown of the data by the class variable.

# # Class Distribution
# 
# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
# 
# 




# class distribution
print(dfcar.groupby('class').size())
 


# In[4]:


#d = {'classe':  ['acc', 'good','unacc','vgood'],'nombredeclasse': dfcar.groupby('class').size() }
#dfmine = pd.DataFrame(data=d) 
 
#pourcentage= [(dfmine.iloc[0,1]/1727)*100, (dfmine.iloc[1,1]/1727)*100,(dfmine.iloc[2,1]/1727)*100,(dfmine.iloc[3,1]/1727)*100]
name = ['acc', 'good','unacc','vgood']
data = [384, 69,1210,65]
explode=(0, 0,0.15, 0)
plt.pie(data,labels=name, explode=explode,autopct='%1.1f%%', startangle=90, shadow=True)
#Les données explode indiquent les espaces entre les parts
#autopct indique le pourcentage réel par rapport aux données qu'on lui a fournit
plt.axis('equal')
plt.show()


# # Data Visualization
# 
# We now have a basic idea about the data. We need to extend that with some visualizations.
# 
# We are going to look at two types of plots:
# 
# Univariate plots to better understand each attribute.
# Multivariate plots to better understand the relationships between attributes.

#  The problem with the above data is it has categorical lablels which is unsuitable for machine learning algorithms. You need to convert them to unique numerical values for machine learning. Lets do it with pandas in python First we will import the csv into pandas

# 
# ('vhigh','high','med','low')=(1,2,3,4) 
#  
#  ('vhigh','high','med','low')=(1,2,3,4) 
#  
#  ('2','3','4','5more')=(1,2,3,4) 
#  
#  ('2','4','more')=(1,2,3) 
#  
#  ('small','med','big')=(1,2,3) 
#  
#  ('low','med','high')=(1,2,3) 
# 
# ('unacc','acc','good','vgood')=(1,2,3,4) 

#  Now our data is ready for machine learning using scikit


dfnum = pd.read_table('car.csv', sep=',', header=0 )
dfnum.head()


# descriptions
print(dfnum.describe())


plt.hist((dfnum.classe))


# acc       384<br>
# good       69<br>
# unacc    1210<br>
# vgood      65<br>

# Samples distributed among 'Classes' have a positive skew, with majority being in the 'unacc'(unacceptable),'acc'(acceptable) output class
# 

# Given that the input variables are numeric, we can create box and whisker plots of each.

# This gives us a much clearer idea of the distribution of the input attributes:


# box and whisker plots
dfnum.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()


# Now we can look at the interactions between the variables.
# 
# 

# Every input feature seems evenly distributed, implying their multivariate interelation is what causes the skew in the 'classe'
# 
#  We can also create a histogram of each input variable to get an idea of the distribution.
# 

# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.
# 
# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.



# histograms
dfnum.hist()
plt.show()


# Multivariate Plots
# 
# Now we can look at the interactions between the variables.
# 
# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.




# scatter plot matrix
scatter_matrix(dfnum)
plt.show()


#class(safety )
sns.pointplot(x="safety", y="classe", data=dfnum);
axes = plt.gca()
axes.set_ylim(0, 4)
  



sns.boxplot(x="maint", y="classe", data=dfnum);


sns.barplot(x="maint", y="classe", data=dfnum);



# # searching for the most accurate model

# Now that our data is numeric, we make setup things for machine learning.
# 
# First we convert from pandas to numpy


car=dfnum.values


# Then we split the data to X,y which is attributes and output class (small y)


X,y = car[:,:6], car[:,6]
 


# Lets split the data for 80% train and 20% test test for scikit machine learning




validation_size = 0.05
seed = 8
X_train, X_test, y_train, y_test =model_selection.train_test_split(X,y,test_size=validation_size,random_state=seed)
seed = 7

scoring = 'accuracy'


# # Build Models
# 
# We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results..<br>
# 
# Let’s evaluate 6 different algorithms:.<br>
# 
# Logistic Regression (LR).<br>
# Linear Discriminant Analysis (LDA).<br>
# K-Nearest Neighbors (KNN)..<br>
# Classification and Regression Trees (CART)..<br>
# Gaussian Naive Bayes (NB)..<br>
# Support Vector Machines (SVM)..<br>
# 
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable..<br>
# 
# Let’s build and evaluate our five models:.<br>



 
# Spot Check Algorithms
#ajouter tous les modèles
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# We can see that it looks like CART has the largest estimated accuracy score.
# 
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation) wihch means run 10 separate learning experiments .


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)

plt.show()


# # Make Predictions

# # KNN


# Make predictions on validation dataset
X,y = car[:,:6], car[:,6]
 
validation_size = 0.2
seed = 8
X_train, X_test, y_train, y_test =model_selection.train_test_split(X,y,test_size=validation_size,random_state=seed)



#the algorithm here does not perform any optimization, but will just save all the data in memory. 
#It's his way of learning in a way.

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# create model
import numpy
import pandas
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import to_categorical


from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm


model = Sequential()
model.add(Dense(25, input_dim=6, init='uniform', activation='relu'))
model.add(Dense(30, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_Train, Y_Train, epochs=600, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_Test, Y_Test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


import numpy as np
from collections import Counter



#deviding predictor and target variables by indexing for both train and test dataset

X_train = car[:,:6]
y_train = car[:,6]

#defining predict function which calculates euclidian distance from test points to all train points
#and returns common most class 

def predict(X_train, y_train, X_test, k):

   
	 # create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
        # first
#(start by calculating the euclidean distance between them other points (class))
		distance = np.sqrt(np.sum(np.square(X_test - X_train[i, :])))
		

#add distances on the list
		distances.append([distance, i])


#(sorted by increasing distances)
	distances = sorted(distances)


# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

# return most common target
	return Counter(targets).most_common(1)[0][0]
predict(X_train, y_train,[1,1,6,5,3,3], 4)


knn.fit(X_train, y_train)
error = 1 - knn.score(X_test, y_test)
print('Erreur: %f' % error)


from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
errors = []
for k in range(1,5):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(X_train, y_train).score(X_test, y_test)))
plt.plot(range(1,5), errors, 'o-')
plt.show()

#Comme on peut le voir, le k-NN le plus performant est celui pour lequel k = 7. On connait donc notre classifieur final optimal
#: 1-nn. Ce qui veut dire que c'est celui qui classifie le mieux les données
# et qui donc dans ce cas précis reconnait au mieux les nombres écrits à la main. 



from sklearn.model_selection import cross_val_score
myList = list(range(1,10))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=4)
    scores = cross_val_score(knn, X_train, y_train, cv=10 ,scoring='accuracy')
    cv_scores.append(scores.mean())
    
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k) 

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

 


# class
# 
# acc=1      
# 
# good=2      
# 
# unacc=3    
# 
# vgood=4    








# # CART









dfnum = pd.read_table('car.csv', sep=',', header=0 )

X_TrainCART,X_TestCART=car[0:1719,:7],car[1720:1727,:7]


training_data= X_TrainCART.tolist()
 

training_data

# Column labels.
# These are used only to print the tree.
header = ["buying","maint","doors","persons","lug_boot","safety","classe"]


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


#######
# Demo:
unique_vals(training_data, 0)
unique_vals(training_data, 0)
# unique_vals(training_data, 1)
#######


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts



#######
# Demo:
class_counts(training_data)
#######


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)



#######
# Demo:
is_numeric(7)
 #######




class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for buying) and a
    'column value' (e.g., 3). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))




#######
# Demo:
# Let's write a question for a numeric attribute
#row 1= maint ,med=3
q=Question(1, 3)
q



def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows



#######
# Demo:
# Let's partition the training data based on whether rows are 3(med).
true_rows, false_rows = partition(training_data, Question(1, 3))
# This will contain all the '3' rows and '4' rows.
true_rows



# This will contain everything else.
false_rows
#######


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


# to build an effective tree you have to know which question to ask and when and to do that we need to quantify how much 
# question helps  to  unmix the labels and we can quantify the amount of uncertainty at single node using a metric called gini impurity
# and we can quantify how much question reduces that unceratinty using a concept gain information
# use this to know the best  question to ask at each point
# divide a data until no further question to ask
# what type of question :iterate over every value for each feature
#     



#######
# Demo:
# Let's look at some example to understand how Gini Impurity works.
#
# First, we'll look at a dataset with no mixing.
no_mixing = [['1'],['1']]
# this will return 0
gini(no_mixing)



# Now, we'll look at dataset with a 50:50 vhigh:high ratio
some_mixing = [[1], [2]]
# this will return 0.5 - meaning, there's a 50% chance of misclassifying
# a random example we draw from the dataset.
gini(some_mixing)


# Now, we'll look at a dataset with many different labels
lots_of_mixing = [[1],
                  [2],
                  [3],
                  [4],
                  ]
# This will return 0.75
gini(lots_of_mixing)
#######


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)



#######
# Demo:
# Calculate the uncertainy of our training data.
current_uncertainty = gini(training_data)
current_uncertainty




# How much information do we gain by partioning on 'high' maint?
#C=0.6 uncertainty of the data
#calculating the imuprity of the child node
#A=avg(uncertainty(number of false rows/total nubmer)+uncertainty(number of true rows/total number)
#info gain(C-A)
true_rows, false_rows = partition(training_data, Question(1, 3))
info_gain(true_rows, false_rows, current_uncertainty)



# How much information do we gain by partioning on 'high' maint?
 
true_rows, false_rows = partition(training_data, Question(5, 2))
info_gain(true_rows, false_rows, current_uncertainty)




# What about if we partioned on 'med' instead?
true_rows, false_rows = partition(training_data, Question(1,2))
info_gain(true_rows, false_rows, current_uncertainty)


# How much information do we gain by partioning on 'high' maint?
#C=0.45 uncertainty of the data
#calculating the imuprity of the child node
#A=avg(number of false rows/total nubmer+number of true rows/total number)
#info gain(C-A)
true_rows, false_rows = partition(training_data, Question(0,3))
info_gain(true_rows, false_rows, current_uncertainty)



# It looks like we learned more using 3 (0.37), than '2(0.14).
# Why? Look at the different splits that result, and see which one
# looks more 'unmixed' to you.
true_rows, false_rows = partition(training_data, Question(5,3)) #max between 60 and 61

# Here, the true_rows contain only 'Grapes'.
true_rows



# the false rows .
false_rows


# On the other hand, partitioning by 3 doesn't help so much.
true_rows, false_rows = partition(training_data, Question(5,2))#min between 60 and 61

# We've isolated one apple in the true rows.
true_rows

# But, the false-rows are badly mixed up.
false_rows
#######


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


#######
# Demo:
# Find the best question to ask first for our toy dataset.
best_gain, best_question = find_best_split(training_data)
best_question
# FYI: is safety== 3 is just as good. See the note in the code above
# where I used '>='.
#######

# How much information do we gain by partioning on '2' safety?
 
true_rows, false_rows = partition(training_data, Question(5, 2))
info_gain(true_rows, false_rows, current_uncertainty)


class Leaf:
#information gain =0
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., unacc) -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch



def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")





my_tree = build_tree(training_data)




#input :training set
#output :refrence to the root node of our tree





print_tree(my_tree)




def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)




#######
# Demo:
# The tree predicts the 1st row of our
# training data is  1with confidence 1.
classify(training_data[1], my_tree)
#######




#######
# Demo:
# The tree predicts the 1st row of our
# training data is  1with confidence 1.
classify(training_data[2], my_tree)
#######




#######
# Demo:
# The tree predicts the 1st row of our
# training data is  1with confidence 1.
classify(training_data[3], my_tree)
#######




#######
# Demo:
# The tree predicts the 1st row of our
# training data is  1with confidence 1.
classify(training_data[4], my_tree)
#######




#######
# Demo:
# The tree predicts the 1st row of our
# training data is  1with confidence 1.
classify(training_data[100], my_tree)
#######



def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs





#######
# Demo:
# Printing that a bit nicer
print_leaf(classify(training_data[1300], my_tree))
#######




#######
# Demo:
# On the second example, the confidence is lower
print_leaf(classify(training_data[100], my_tree))
#######




testing_data=X_TestCART





unique_vals(testing_data, 0)




for row in testing_data:
    print ("Actual: %s. Predicted: %s" %
           (row[-1], print_leaf(classify(row, my_tree))))
