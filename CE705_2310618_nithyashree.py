#importing numpy as it useful for 2d array handling and mathematical operations 
import numpy as np
#importing CSV to read the file
import csv

class matrix:
    def __init__(self, filename=None):
        """
        whenever a new class is created the init method is used to set up the object and load the data
        
        Parameters:
        filename as none which means it is an optional parameter and can create a matrix object without a filename
        """

        #assign array_2d to none
        self.array_2d = None

        # If the filename is provided, the load_from_csv method  will be called to load the data from the file
        if filename:
            self.load_from_csv(filename)
    
    def load_from_csv(self, filename):
        """
        This method is created to load the data from a CSV file
        
        Parameters: filename (as per assignment instruction this method should have one parameter)
        """

        #opens the CSV file and reads the CSV file
        with open(filename, 'r') as file_read:
            #csv.reader handles the comma and spaces while file reading 
            func_read = csv.reader(file_read)

            #reads all the data and converts it to a NumPy array with the datatype float and stores it in the array_2d
            self.array_2d = np.array(list(func_read), dtype=float)
    
    def standardise(self):
        """
        This method standardizes the array_2d using the formula (stand_ard - stand_ard_mean) / (stand_ard_max - stand_ard_min)

        Parameters: None

        """
        standard_array2d = self.array_2d
        #calculates the mean of each column in the array and axis = 0 will make Numpy calculate along columns
        stand_array2d_mean = np.mean(standard_array2d, axis=0)
        #calculates the maximum of each column 
        stand_array2d_max = np.max(standard_array2d, axis=0)
        #calculates the minimum
        stand_array2d_min = np.min(standard_array2d, axis=0)
        #standarising the array using the formula
        self.array_2d = (standard_array2d - stand_array2d_mean) / (stand_array2d_max - stand_array2d_min)
    
    def get_distance(self, other_matrix, row_i):
        """
        This method calculates the Euclidean distance between a (row_i) in this matrix and each rows of another matrix (other_matrix).

        Parameters:  other_matrix -> Another matrix object whose rows are compared to the row_i of this matrix.
                      row_i -> The index of the row in the current matrix

        Returns: This method returns the calculated Euclidean distance of the matrix which has n rows and 1 columns
        """
        #get the row_i from the current matrix
        row_i_ofdata = self.array_2d[row_i]
        #calculating the Euclidean distance 
        distance_matrix = np.sum((other_matrix.array_2d - row_i_ofdata)**2, axis=1)
        #Reshape the distance array to make sure the result has n rows and 1 column and then store it in a new matrix object
        result_imatrix = matrix().get_from_2darray(distance_matrix.reshape(-1, 1))
        #returing the matrix of distance (n rows and one column)
        return result_imatrix

    def get_weighted_distance(self, other_matrix, weights, row_i):
        """
        This method calculates the weighted Euclidean distance between a row (row_i) in this matrix and each row in another matrix (other_matrix).

        Parameters: other_matrix -> Another matrix 
                    weights -> a matrix weight
                    row_i -> The index of the row in the current matrix

        Returns: This method returns the calculated weighted Euclidean distance of the matrix which has n rows and 1 columns
        """
        # Extract the row_i from current matrix
        the_row_i_ofdata = self.array_2d[row_i]
        # Performing weighted Euclidean distance as per the formula
        diff_matrix = other_matrix.array_2d - the_row_i_ofdata
        weightmatrix_squrd_diff = weights.array_2d * (diff_matrix ** 2)
        dist_weight = np.sum(weightmatrix_squrd_diff, axis=1)
        # creates a new matrix which has n rows and 1 column
        result_weightmatrix = matrix().get_from_2darray(dist_weight.reshape(-1, 1))
        # returns the matrix which has Weighted Euclidean distance
        return result_weightmatrix

    def get_count_frequency(self):
        """
        This method calculates the frequency of unique elements in the column

        Parameters: None

        Returns: This method returns 0 if the number of columns in array_2d is not one 
                 Also, the method returns the count of unique elements

        """

        #check if array_2d has more than one column, if so return 0
        if self.array_2d.shape[1] != 1:
            return 0
        # Calculate the count of unique elements in the single column
        unique_entries, count_entries = np.unique(self.array_2d, return_counts=True)
        #returns dictionary with key and value pairs for a single column
        return {int(unidata): int(countdata) for unidata, countdata in zip(unique_entries, count_entries)}

    def get_from_2darray(self, array_numpy):
        """
        This method is used to create new matrix object from a existing array

        Parameters: array_numpy -> used to initialize array

        Returns: This method returns a new matrix object

        """

        #create new matrix object
        new_matrixobj = matrix()
        #set the array_2d attribute of the new matrix to the input 
        new_matrixobj.array_2d = array_numpy
        #return a new matrix
        return new_matrixobj


#FUNCTION

def get_initial_weights(m):
    """
    This function is used to get random initial weights with one row and m columns

    Parameters: m -> the number of columns (weights) in the matrix

    Returns: This function returns the matrix object with one row and m columns of random initial weights

    """
    # get initial random weights
    initial_weights = np.random.rand(1, m)
    # Normalize the weights so that the sum of all values equals 1
    normalinit_weights = initial_weights / np.sum(initial_weights)
    #create a matrix object from normalized weight
    res_initweight = matrix().get_from_2darray(normalinit_weights)
    # Return the matrix object
    return res_initweight

def get_centroids(matrix_containing_data, S, K):
    """
    This function calculates the  centroids for each cluster based on the current cluster assignments (step 9)

    Parameters: matrix_containing_data -> This parameter contains the data points
                S -> The matrix containing cluster for each data point
                K -> no of clusters

    Returns: This function returns the matrix object containing K(no of clusters) rows and the same number of columns as the data matrix.
    """

    #Initialize an empty list to store the centroids
    centroids_main = []
    # loop through each cluster
    for k in range(1, K + 1):
        #select the data points assigned to cluster k based on the S matrix
        cluster_datacentroid = matrix_containing_data.array_2d[S.array_2d.flatten() == k]
        #if the data points in the cluster are greater than zero calculate centroids
        if len(cluster_datacentroid) > 0:
            centroid_matrix = np.mean(cluster_datacentroid, axis=0)
        else:
            # If it is empty assign it as zero
            centroid_matrix = np.zeros(matrix_containing_data.array_2d.shape[1])
         # Add the calculated centroid to the list centroid_main   
        centroids_main.append(centroid_matrix)
    # Convert the list of centroids to a NumPy array and create a matrix object
    centroid_results= matrix().get_from_2darray(np.array(centroids_main))
    # return the list of centroids to a matrix object
    return centroid_results

def get_separation_within(matrix_containing_data, centroids_containing_matrix, S, K):
    """
    This function calculates the separation within clusters.
    
    Parameters: matrix_containing_data ->  matrix containing the data points
                centroids_containing_matrix -> matrix containing the centroids
                S -> matrix containing cluster assignments
                K -> number of clusters
    
    Returns:  This function returns a matrix with 1 row and m columns, containing the separation within clusters
    """
    # get the no of columns in the data
    num_of_columns = matrix_containing_data.array_2d.shape[1]
    #initialize an array to store the separation within clusters
    separation_within_func = np.zeros(num_of_columns)
    
    # loop through each cluster k
    for k in range(1, K + 1):
        # extract the indices of data points assigned to cluster k
        cluster_within_indices = np.where(S.array_2d.flatten() == k)[0]

        #if the data points in the cluster are greater than zero
        if len(cluster_within_indices) > 0:
            # get the data points of the current cluster
            cluster_within_data = matrix_containing_data.array_2d[cluster_within_indices]
            #get the centroid of the cluster K
            centroid_within = centroids_containing_matrix.array_2d[k-1]
            # Calculayte  the sum of squared distances for each feature
            separation_within_func += np.sum((cluster_within_data - centroid_within) ** 2, axis=0)

    #create a matrix object containing the separation within clusters 
    result_withindistance = matrix().get_from_2darray(separation_within_func.reshape(1, -1))
    # Return the separation within clusters as a matrix object
    return result_withindistance

def get_separation_between(matrix_containing_data, centroids_containing_matrix, S, K):
    """
    This function calculates the separation between the clusters 

    Parameters: matrix_containing_data -> matrix containing the data points
                centroids_containing_matrix ->  matrix containing the centroids of the data points
                S -> matrix containing cluster assignments of the data points
                K -> number of clusters 

    Returns:  This function returns a matrix object  with 1 row and m columns, containing the separation between clusters
    """
 
   # Calculate the overall mean of the data matrix
    all_mean_cluster = np.mean(matrix_containing_data.array_2d, axis=0)
    
    # initialize a vector to store the separation between clusters
    num_of_columns = matrix_containing_data.array_2d.shape[1]
    separation_between_cluster = np.zeros(num_of_columns)
    
    # loop through each cluster k
    for k in range(1, K + 1):
        # Number of data points in cluster k
        size_of_cluster = np.sum(S.array_2d == k)
        
        # Calculate the squared Euclidean distance between each centroid and the overall mean
        centroid_between = centroids_containing_matrix.array_2d[k-1]
        separation_between_cluster += size_of_cluster * ((centroid_between - all_mean_cluster) ** 2)
    
    # Create a matrix object
    between_results = matrix().get_from_2darray(separation_between_cluster.reshape(1, -1))
    
    # Return the separation between clusters as a matrix object
    return between_results


def get_groups(matrix_containing_data, K):
    """
    This function performs the clustering algorithm steps by using the previous methods and functions
    
    Parameters: matrix_containing_data -> Matrix object containing the data points
                K -> Number of groups to be created
    Returns: This function returns matrix S with n rows and 1 column, containing cluster assignments
    """
    # As per More details in the assignment performing parameter validation for K before starting the algorithm for cluster formation
    n = matrix_containing_data.array_2d.shape[0]
    if not 2 <= K <= n-1:
        raise ValueError(f"K must be in the range [2, {n-1}]")
        
    # Step 2 -> Initialize the matrix weight with 1 row and m column
    #get the number of columns in the matrix containing data
    num_of_features = matrix_containing_data.array_2d.shape[1]
    #use the get_initial_weights method to achieve Step 2
    weights_initialgroup = get_initial_weights(num_of_features)
    
    # Step 3 & 4 -> create centroids and set S to zero
    #extract the no of rows
    no_of_rows = matrix_containing_data.array_2d.shape[0]
    # an empty matrix to store centroid
    centroids_groups = matrix()
    #set S to zero initially to store clusters with n rows and 1 columns
    S = matrix().get_from_2darray(np.zeros((no_of_rows, 1)))
    
    # Step 5 & 6 -> Select K different rows  from the data matrix randomly and copy to matrix centroids
    #selecting k different rows randomly
    random_rows_group = np.random.choice(no_of_rows, K, replace=False)
    #copying random values to matrix centroids
    centroids_groups.array_2d = matrix_containing_data.array_2d[random_rows_group]
    # To maintain stable iteration of clusters and check for convergence this loop continues until the cluster formation stops changing,
    while True: 
        # Store the current cluster to compare with the updated ones
        S_previous = S.array_2d.copy()
        
        # Step 7 -> Assign each data point to the nearest centroid
        for i in range(no_of_rows):
            #using get_weighted_distances to calculate the distance of data
            distances_groups = matrix_containing_data.get_weighted_distance(centroids_groups, weights_initialgroup, i)
            # Assign data  i to the cluster with the nearest centroid and add 1 to convert from 0-based to 1-based indexing
            S.array_2d[i, 0] = np.argmin(distances_groups.array_2d) + 1
        
        # Step 8 -> Check if S has changed 
        if np.array_equal(S.array_2d, S_previous):
            break
        
        # Step 9 -> Update centroids using get_centroid method
        centroids_groups = get_centroids(matrix_containing_data, S, K)
        
        # Step 10 -> Update weights
        weights_initialgroup = get_new_weights(matrix_containing_data, centroids_groups, weights_initialgroup, S, K)
    
    return S

def get_new_weights(matrix_containing_data, centroids_containing_matrix, old_weightvector, S, K):
    """
    This function calculates the new  matrix weights for each features in the clustering algorithm

    Parameters -> matrix_containing_data -> matrix containing the data points
                centroids_containing_matrix ->  matrix containing the centroids of the data points
                old_weightvector -> the weight to be updated in the matrix
                S -> matrix containing cluster assignments of the data points
                K -> number of clusters 
    Returns -> This function returns a new matrix of updated weights with 1 row and many columns
    """
    # Get the number of features
    num_of_features = matrix_containing_data.array_2d.shape[1]
    
    # Calculate separation within and between clusters using the previous method
    separation_within_weights = get_separation_within(matrix_containing_data, centroids_containing_matrix, S, K)
    separation_between_weights = get_separation_between(matrix_containing_data, centroids_containing_matrix, S, K)
    
    # Initialize new weights array to store new weights
    new_weights_vector = np.zeros(num_of_features)
    
    # Calculate the sum of (separation_between / separation_within) for normalization
    sum_btwn_div_within = np.sum(separation_between_weights.array_2d / separation_within_weights.array_2d)
    
    # Update each weight through the loop
    for i in range(num_of_features):
        # check if separation_within_weights is not zero to avoid division by zero
        if separation_within_weights.array_2d[0, i] != 0:  
            #calculate the new weight using the formula
            new_weights_vector[i] = 0.5 * (old_weightvector.array_2d[0, i] + 
                                    (separation_between_weights.array_2d[0, i] / separation_within_weights.array_2d[0, i]) / sum_btwn_div_within)
        else:
            #if it is zero updates to old weight
            new_weights_vector[i] = old_weightvector.array_2d[0, i] 
            
    # Create and return a new matrix object with the updated weights       
    result_weightget = matrix().get_from_2darray(new_weights_vector.reshape(1, -1))

    return result_weightget


def run_test():
    # Test function to run the clustering on 'Data.csv' for K=2 to 10
    m = matrix('/Users/nithyashree/Desktop/Python Reassessment/CE705 Reassessment 2024/Data.csv')
    m.standardise()
    for k in range(2, 11):
        for i in range(20):
            S = get_groups(m, k)
            print(f"{k}={S.get_count_frequency()}")

if __name__ == "__main__":
    run_test()
