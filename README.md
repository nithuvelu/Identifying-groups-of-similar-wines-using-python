### Assignment: identifying groups of similar wines
A sommelier is a trained professional who spends his or her day tasting different wines, and identifying similarities (or sometimes dissimilarities) between these. Given this is clearly an exhausting task, you have been hired to develop a software capable of grouping similar wines together. Your software will load a data set containing information about each wine (Alcohol content, alkalinity of ash, Proanthocyanins, colour intensity, etc) and identify which wines are similar.
Luckily, your employer has already identified a suitable algorithm and designed the software for you. All you are required to do is to write the actual source code (with comments).
Technical details:
You’ll be using different data structures to accomplish the below. Your assignment must contain the code for the functions and methods below. If you wish you can write more functions and methods, but those described below must be present.
ATTENTION: in the text below any mention of matrix should be read as relating to the Class matrix you have to write (see below), and not some other matrix that may exist in some python module. Any parameters of methods or functions must be in the order presented below.
1) Class: matrix
You will code a class called matrix, which will have an attribute that must be called array_2d. This attribute is supposed to be a NumPy array containing numbers in two dimensions. The class matrix must have the following methods:
(in these, the parameters are in addition to self and must be used in the order presented below)
load_from_csv
This method should have one parameter, a file name (including, if necessary, its path and extension). This method should read this CSV file and load its data to the array_2d of matrix. Each row in this file should be a row in array_2d. Notice that in CSV files a comma separates columns (CSV = comma separated values).
You should also write code so that
m = matrix(‘validfilename.csv’)
Creates a matrix m with the data in the file above in array_2d.
standardise
This method should have no parameters. It should standardise the array_2d in the matrix calling this method. For details on how to standardise a matrix, read the appendix.
get_distance
This method should have two parameters: a matrix (let us call it other_matrix), and a row number (let us call it row_i). This method should return a matrix containing the Euclidean distance between the row row_i of the matrix calling this method and each of the rows in other_matrix. For details about how to calculate this distance, read the appendix.
To be clear: if other_matrix has n rows, the matrix returned in this method will have n rows and 1 column.
get_weighted_distance
This method should have three parameters: two objects of matrix (let us call them other_matrix, and weights), and a row number (let us call it row_i). This method should return a matrix containing the Weighted Euclidean distance between the row row_i of the matrix calling this method and each of the rows in other_matrix, using the weights in the matrix weights. For details about how to calculate this distance, read the appendix.
To be clear: if other_matrix has n rows, the matrix returned in this method will have n rows and 1 column.
get_count_frequency
This method should have no parametes, and it should work if the array_2d of the matrix calling this method has only one column. This method should return a dictionary mapping each element of the
  4
array_2d to the number of times this element appears in array_2d. If the number of columns in array_2d is not one, then this method should return the integer 0.
2) Functions
The code should also have the functions (i.e. not methods, so not part of the class matrix) below. No code should be outside any function or method in this assignment (the only exception are imports).
get_initial_weights
This function should have one parameter, an integer m. This function should return a matrix with 1 row and m columns containing random values, each between zero and one. The sum of these m values must be equal to one.
get_centroids
This function should have three parameters: (i) a matrix containing the data, (ii) the matrix S, (iii) the value of K. This function should implement the Step 9 of the algorithm described in the appendix. It should return a matrix containing K rows and the same number of columns as the matrix containing the data.
get_separation_within
This function should have four parameters: a matrix containing the data, a matrix containing the centroids, the matrix S, and number of groups to be created (K). This function should return a matrix with 1 row and m columns (m is the number of columns in the matrix containing the data), containing the separation within clusters (see the Appendix for details).
get_separation_between
This function should have four parameters: a matrix containing the data, a matrix containing the centroids, the matrix S, and number of groups to be created (K). This function should return a matrix with 1 row and m columns (m is the number of columns in the matrix containing the data), containing the separation between clusters (see the Appendix for details).
get_groups
This function should have two parameters: a matrix containing the data, and the number of groups to be created (K). This function follows the algorithm described in the appendix. It should return a matrix S (defined in the appendix). This function should use the other functions you wrote as much as possible. Do not keep repeating code you already wrote.
get_new_weights
This function takes five parameters: (i) a matrix containing the data, (ii) a matrix containing the centroids, (iii) a matrix containing the weights to be updated (this is referred as old weight vector in the Appendix), (iv) the matrix S, (v) the number of groups to be created (K). This function should return a new matrix weights with 1 row and as many columns as the matrix containing the data (and the matrix containing the centroids). This function should use the other functions you wrote as much as possible. Follow Step 10 of the algorithm in the Appendix.
run_test
Your code must contain the function below. Do not change anything, I’ll have a file called Data.csv in the same folder as your code.
def run_test():
   m = matrix(‘Data.csv’)
   for k in range(2,11):
for i in range(20):
S = get_groups(m, k) print(str(k)+‘=’+str(S.get_count_frequency()))
The aim of this function is just to run a series of tests. By consequence, here (and only here) you can use the hard-coded values above for the strings containing the filenames of data and values for K.
