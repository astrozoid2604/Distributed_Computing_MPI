import numpy as np
import pandas as pd
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from time import time, gmtime, strftime
from tqdm import tqdm



# Globals
test_size          = 0.3
input_features     = [#'latitude','longitude' #These 2 columns are just unique identifiers and not input features
                      'housingMedianAge','totalRooms','totalBedrooms','population','households','medianIncome',
                     ]
nominal_feature    = "oceanProximity"
target_feature     = "medianHouseValue"
unique_identifiers = ['latitude', 'longitude']



def standardize_data(X_train, X_test):
    """
    Standardize an input feature so that the data distribution follows Normal distribution ~ N(0, 1)
    This is done to achive faster convergence.

    Parameters:
    column: np.ndarray of shape(n_samples,)

    Returns:
    ans: np.ndarray, standardized input features of an input feature of shape(n_samples,)
    """
    
    scaler = StandardScaler()
    
    # Reshape the column to fit the scaler and standardize
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

    
def rbf_kernel(X1, X2, **kwargs):
    """
    Compute the Gaussian (RBF) kernel between two matrices.
    
    Parameters:
    X1: np.ndarray of shape (n_samples_1, n_features)
    X2: np.ndarray of shape (n_samples_2, n_features)
    gamma: float, kernel coefficient
    
    Returns:
    K: np.ndarray of shape (n_samples_1, n_samples_2)
    """
    
    # Extract gamma from kwargs, set a default value if not provided
    gamma = kwargs.get('gamma', 1.0)

    # Compute squared Euclidean distances between each pair of points
    sq_dists = -2 * np.dot(X1, X2.T) + np.sum(X2**2, axis=1) + np.sum(X1**2, axis=1)[:, np.newaxis]
    
    # Compute the RBF kernel matrix
    K = np.exp(-gamma * sq_dists)
    return K


def linear_kernel(X1, X2, **kwargs):
    """
    Compute the Linear kernel between two matrices.
    
    Parameters:
    X1: np.ndarray of shape (n_samples_1, n_features)
    X2: np.ndarray of shape (n_samples_2, n_features)
    kwargs: dict, Additional keyword arguments (not used here)
    
    Returns:
    K: np.ndarray of shape (n_samples_1, n_samples_2)
    """

    # Compute the Linear kernel matrix
    K = np.dot(X1, X2.T)
    return K


def polynomial_kernel(X1, X2, **kwargs):
    """
    Compute the Polynomial kernel between two matrices.
    
    Parameters:
    X1: np.ndarray of shape (n_samples_1, n_features)
    X2: np.ndarray of shape (n_samples_2, n_features)
    kwargs: dict, Additional keyword arguments including 'degree' and 'coef0'
    
    Returns:
    K: np.ndarray of shape (n_samples_1, n_samples_2)
    """

    # Extract degree and coef0 from kwargs, set a default value if not provided
    degree = kwargs.get('degree', 3)
    coef0 = kwargs.get('coef0', 1)

    # Compute the Polynomial kernel matrix
    K = (np.dot(X1, X2.T) + coef0) ** degree
    return K


def sigmoid_kernel(X1, X2, **kwargs):
    """
    Compute the Sigmoid kernel between two matrices.
    
    Parameters:
    X1: np.ndarray of shape (n_samples_1, n_features)
    X2: np.ndarray of shape (n_samples_2, n_features)
    kwargs: dict, Additional keyword arguments including 'gamma' and 'coef0'
    
    Returns:
    K: np.ndarray of shape (n_samples_1, n_samples_2)
    """

    # Extract gamma and coef0 from kwargs, set a default value if not provided
    gamma = kwargs.get('gamma', 0.1)
    coef0 = kwargs.get('coef0', 0)


    # Compute the Sigmoid kernel matrix
    K = np.tanh(gamma * np.dot(X1, X2.T) + coef0)
    return K


def laplacian_kernel(X1, X2, **kwargs):
    """
    Compute the Laplacian kernel between two matrices.
    
    Parameters:
    X1: np.ndarray of shape (n_samples_1, n_features)
    X2: np.ndarray of shape (n_samples_2, n_features)
    kwargs: dict, Additional keyword arguments including 'gamma'
    
    Returns:
    K: np.ndarray of shape (n_samples_1, n_samples_2)
    """

    # Extract gamma from kwargs, set a default value if not provided
    gamma = kwargs.get('gamma', 1.0)

    # Compute Manhattan distances between each pair of points
    dist = np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)

    # Compute the Laplacian kernel matrix
    K = np.exp(-gamma * dist)
    return K

    
def kernel_ridge_regression(K, y, alpha):
    """
    Perform Kernel Ridge Regression.
    
    Parameters:
    K: np.ndarray, Kernel matrix of shape (n_samples, n_samples)
    y: np.ndarray, Target values of shape (n_samples,)
    alpha: float, Regularization parameter
    
    Returns:
    alpha_hat: np.ndarray, Regression coefficients of shape (n_samples,)
    """
    
    n_samples = K.shape[0]
    I = np.eye(n_samples)  # Identity matrix for regularization
    
    # Solve the regularized linear system
    alpha_hat = np.linalg.solve(K + alpha * I, y)
    return alpha_hat


def predict(K, alpha_hat):
    """
    Make predictions using the trained KRR model.
    
    Parameters:
    K: np.ndarray, Kernel matrix between training and test data of shape (n_test_samples, n_train_samples)
    alpha_hat: np.ndarray, Regression coefficients of shape (n_samples,)
    
    Returns:
    y_pred: np.ndarray, Predicted values of shape (n_test_samples,)
    """
    
    # Compute predictions as dot product of kernel matrix and coefficients
    return np.dot(K, alpha_hat)


def one_hot_encode(df, nominal_columns):
    """
    One-hot encode the nominal features in the dataframe.
    
    Parameters:
    df: pd.DataFrame, Input dataframe
    nominal_columns: list, List of nominal feature names
    
    Returns:
    df_encoded: pd.DataFrame, Dataframe with one-hot encoded nominal features
    """
    
    df_encoded = pd.get_dummies(df, columns=nominal_columns)
    return df_encoded


def parallel_kernel_matrix_computation(X, kernel_base, verbose, **kwargs):
    """
    Compute the global kernel matrix in parallel using MPI.
    
    Parameters:
    X: np.ndarray, Input data of shape (n_samples, n_features)
    gamma: float, Kernel coefficient
    
    Returns:
    K: np.ndarray, Global kernel matrix of shape (n_samples, n_samples)
    """
    comm = MPI.COMM_WORLD        # Initialize MPI communicator
    rank = comm.Get_rank()       # Get the rank (ID) of the current process
    size = comm.Get_size()       # Get the total number of processes

    n_samples = X.shape[0]       # Total number of samples in the input data
    local_n = n_samples // size  # Number of samples each process will handle
    if verbose: print(f"Rank {rank}: n_samples={n_samples}, local_n={local_n}...\n")
    
    # Each process gets a portion of the data based on its rank
    local_X = X[rank * local_n: (rank + 1) * local_n]
    if verbose: print(f"Rank {rank}: Finished extraction subset sample data as local_X with shape {local_X.shape}...\n")

    # Allocate space for the local portion of the kernel matrix
    local_K = np.empty((local_n, n_samples), dtype=float) 
    if verbose: print(f"Rank {rank}: Finished allocating memory for local kernel matrix local_K with shape {local_K.shape}...\n")

    requests = []

    for i in range(size):
        if verbose: print(f"Rank {rank}: Process iteration i={i} of num processor {size}...\n")
        if i == rank:
            # Compute the base kernel matrix for the local data against itself
            local_K[:, rank * local_n: (rank + 1) * local_n] = kernel_base(local_X, local_X, **kwargs)
            if verbose: print(f"Rank {rank} i {i}: Finished computing local_K for local_X against itself at column {rank * local_n} to {(rank + 1) * local_n - 1}...\n")
            
            # Send local data to all other processes using non-blocking send
            for j in range(size):
                if j != rank:
                    req = comm.Issend(local_X, dest=j, tag=rank)
                    requests.append(req)
                    if verbose: print(f"Rank {rank} i {i}: Initiating non-blocking send of local_X to destination processor j={j} with tag {rank}...\n")
        else:
            # Allocate space to receive the data from other processes
            other_X = np.empty((local_n, X.shape[1]), dtype=float)
            
            # Receive the data from process i
            comm.Recv(other_X, source=i, tag=i)
            if verbose: print(f"Rank {rank} i {i}: Finished blocking receive other_X from source processor {i} with tag {i}...\n")
            
            # Compute the base kernel matrix for the local data against the received data
            local_K[:, i * local_n: (i + 1) * local_n] = kernel_base(local_X, other_X, **kwargs)
            if verbose: print(f"Rank {rank} i {i}: Finished computing local_K for local_X against other_X at column {i * local_n} to {(i + 1) * local_n - 1}...\n")
    
    # Wait for all non-blocking sends to complete
    MPI.Request.Waitall(requests)
    if verbose: print(f"Rank {rank}: Finished waiting for all non-blocking send...\n")
    
    # Synchronize all processes
    comm.Barrier()
    if verbose: print(f"Rank {rank}: Finished synchronizing all processes using Barrier()...\n")

    # Allocate space for the full kernel matrix on the root process
    K = None
    if rank == 0:
        K = np.empty((n_samples, n_samples), dtype=float)
        if verbose: print(f"Rank {rank}: Finished allocating space for global kernel matrix on the root process K with shape {K.shape}...\n")

    # Gather the local kernel matrix portions into the full matrix on the root process
    comm.Gather(local_K, K, root=0)
    if verbose: print(f"Rank {rank}: Finished gathering local kernel matrix local_K to global kernel matrix at root process K...\n")

    return K


def compute_rmse(y_pred, y_true):
    """
    Compute RMSE

    Parameters:
    y_pred: np.ndarray, predicted target feature based on K_test
    y_true: np.ndarray, ground-truth target feature

    Returns:
    ans: float, RMSE
    """

    ans = np.sqrt(np.mean((y_pred - y_true)**2))
    return ans


def main():
    comm = MPI.COMM_WORLD  # Initialize MPI communicator
    rank = comm.Get_rank()  # Get rank of current process
    size = comm.Get_size()       # Get the total number of processes
    print(f"Rank of current process: {rank} with total size of {size}...\n")

    # Load and feature engineer data only on root process
    if rank == 0:
        ## Load raw input data
        #df = pd.read_csv('cal_housing.data') # CSV format obtained from this link: https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz
        df = pd.read_csv('housing_with_header.tsv', sep='\t')
        print(f"Rank {rank}: Finished loading housing.tsv file...\n")
        
        # Standardize input features & convert to numpy. 
        # Target feature should not be standardized or otherwise there will be additional effort to convert back to the correct scaling.
        print(f"df.columns={df.columns}")


        y  = df[target_feature]
        if nominal_feature in df.columns:
            # Convert nominal_feature column to one-hot-encoded columns
            nominal_array             = df[nominal_feature].values.reshape(-1, 1)
            num_unique_nominal_values = len(set(df[nominal_feature].values.tolist()))
            encoded_array             = OneHotEncoder().fit_transform(nominal_array)
            nominal_features          = [nominal_feature+"_"+str(i) for i in range(num_unique_nominal_values)]
            df[nominal_features]      = pd.DataFrame(encoded_array.toarray())
            del df[nominal_feature]
            print(f"Rank {rank}: Finished converting nominal feature column '{nominal_feature}' to one-hot encoding...\n")

            X = df[input_features+nominal_features]
            print(f"Rank {rank}: Finished retrieving input features which also contains nominal feature called {nominal_feature}...\n")
        else:
            X = df[input_features]
            print(f"Rank {rank}: Finished retrieving input features. No nominal feature in input features...\n")


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        X_train[input_features], X_test[input_features] = standardize_data(X_train[input_features], X_test[input_features])
            
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy().flatten()
        y_test = y_test.to_numpy().flatten()
        print(f"Rank {rank}: Printing various preprocessed data shape...\nX_train.shape: {X_train.shape}\nX_test.shape: {X_test.shape}\ny_train.shape:{y_train.shape}\ny_test.shape: {y_test.shape}\n")
        print(f"Rank {rank}: Finished loading, standardizing, and splitting train & test data...\n")

        if size > 1:
            # Trimming X_train and y_train to ensure they can be partitioned into equally-sized data block for each processor.
            n_samples = X_train.shape[0]
            remainder = n_samples % size
            X_train = X_train[:-remainder]
            y_train = y_train[:-remainder]
            print(f"Rank {rank}: Finished trimming odd-sized training samples, n_samples={n_samples}, remainder={remainder}, X_train.shape={X_train.shape}, y_train.shape={y_train.shape}...\n")

    else:
        X_train = y_train = X_test = y_test = None
        print(f"Rank {rank}: Finished initializing train & test data...\n")


    # Broadcast feature engineered data to all processes
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    X_test  = comm.bcast(X_test,  root=0)
    y_test  = comm.bcast(y_test,  root=0)
    print(f"Rank {rank}: Finished broadcasting feature engineered data from root to all processes...\n")

    overall_best = {"best_rmse_train":np.inf, "best_rmse_test":np.inf, "best_kernel":"None", "best_params":"None"}

    print(f"{'#--------------------------------------------------------------------------------------------------------------' if rank==0 else ''}\n")

    ###################################################################
    ######### START : Hyper Parameter Tuning for RBF Kernel ###########
    ######### --------------------------------------------- ###########
    ######### Per 09 June 2024, best_gamma = 8.00e-2        ###########
    ######### Per 09 June 2024, best_alpha = 1.40e-2        ###########
    ######### Per 09 June 2024, best_rmse_train= 55795.5797 ###########
    ######### Per 09 June 2024, best_rmse_test = 58488.9359 ###########
    ###################################################################
    kernel_name = "RBF Kernel"
    kernel_base = rbf_kernel # Hyper parameters: gamma, alpha
    verbose     = False

    ## Coarse-Grained HyperParameter Tuning Range
    #gammas      = [1e-7*(10**i) for i in range(1,  7)]  # Gamma is a Kernel coefficient for RBF Kernel. 
    #alphas      = [1e-6*(10**i) for i in range(1,  9)]  # Alpha is a Regularization parameter for RBF Kernel

    ## Fine-Grained HyperParameter Tuning Range
    gammas      = [1e-2 + 1e-2*i for i in range(10)]  # Gamma is a Kernel coefficient for RBF Kernel. # In mathematical formula of KRR with RBF kernel, Gamma is also equal to 0.5/sigma**2 where sigma is width of Gaussian Kernel.
    alphas      = [5e-3 + 1e-3*i for i in range(10)]  # Alpha is a Regularization parameter for RBF Kernel

    best_gamma  = 1.0
    best_alpha  = 1.0
    best_rmse_test   = np.inf
    combinations= [(gamma, alpha) for gamma in gammas for alpha in alphas]
    for gamma, alpha in tqdm(combinations, desc=kernel_name+" Tuning...", ascii=False, ncols=70):
        # Compute kernel matrix in parallel
        K_train = parallel_kernel_matrix_computation(X_train, kernel_base, verbose, gamma=gamma) 

        if rank==0:
            if verbose: print(f"Rank {rank}: Finished constructing global kernel matrix K_train of shape {K_train.shape}...\n")
        
            # Perform Kernel Ridge Regression on root process
            alpha_hat = kernel_ridge_regression(K_train, y_train, alpha)
            if verbose: print(f"Rank {rank}: Finished computing alpha_hat...\n")

            # Compute test kernel matrix
            K_test = kernel_base(X_test, X_train, gamma=gamma)
            if verbose: print(f"Rank {rank}: Finished calculating global test kernel matrix K_test of shape {K_test.shape} based on X_test of shape {X_test.shape} and X_train of shape {X_train.shape}...\n")

            # Make predictions
            y_pred_train= predict(K_train,alpha_hat)
            y_pred_test = predict(K_test, alpha_hat)
            if verbose: print(f"Rank {rank}: Finished inference to obtain y_pred_train {y_pred_train.shape} y_pred_test {y_pred_test.shape}...\n")
            
            # Evaluate RMSE (Root Mean Squared Error)
            rmse_train= compute_rmse(y_pred_train, y_train)
            rmse_test = compute_rmse(y_pred_test,  y_test)
            if verbose: print(f"\nRank {rank} gamma {gamma:.2e} alpha {alpha:.2e}: Finished computing RMSE_train={rmse_train:.4f} RMSE_test={rmse_test:.4f}...\n")

            if rmse_test < best_rmse_test:
                best_gamma = gamma 
                best_alpha = alpha 
                best_rmse_train = rmse_train  
                best_rmse_test  = rmse_test   

            if rmse_test < overall_best["best_rmse_test"]: 
                overall_best["best_rmse_train"]  = rmse_train
                overall_best["best_rmse_test"]   = rmse_test
                overall_best["best_kernel"] = kernel_name
                overall_best["best_params"] = f"gamma:{gamma:.2e}, alpha:{alpha:.2e}"

    if rank==0: print(f"\n{kernel_name} ==> Best gamma {best_gamma:.2e}, Best alpha {best_alpha:.2e}, Best RMSE_train {best_rmse_train:.4f}, Best RMSE_test {best_rmse_test:.4f}...\n")
    ###################################################################
    ######### FINISH: Hyper Parameter Tuning for RBF Kernel ###########
    ###################################################################

    print(f"{'#--------------------------------------------------------------------------------------------------------------' if rank==0 else ''}\n")

    ###################################################################
    ######### START : Hyper Parameter Tuning for Laplacian Kernel #####
    ######### ---------------------------------------------------- ####
    ######### Per 09 June 2024, best_gamma  = 1.20e-1              ####
    ######### Per 09 June 2024, best_alpha  = 2.00e-1              ####
    ######### Per 09 June 2024, best_rmse_train= 50232.4180        ####
    ######### Per 09 June 2024, best_rmse_test = 59256.2770        ####
    ###################################################################
    kernel_name = "Laplacian Kernel"
    kernel_base = laplacian_kernel # Hyper parameters: gamma, alpha
    verbose     = False

    ## Coarse-Grained HyperParameter Tuning Range
    #gammas      = [1e-7*(10**i) for i in range(1,  7)]  # Gamma is a Kernel coefficient for RBF Kernel. 
    #alphas      = [1e-6*(10**i) for i in range(1,  9)]  # Alpha is a Regularization parameter for RBF Kernel

    ## Fine-Grained HyperParameter Tuning Range
    gammas      = [3e-2   + 1e-2*i for i in range(10)]  # Gamma is a Kernel coefficient for Laplacian Kernel. 
    alphas      = [1.1e-1 + 1e-2*i for i in range(10)]  # Alpha is a Regularization parameter for Laplacian Kernel

    best_gamma  = 1.0
    best_alpha  = 1.0
    best_rmse_test   = np.inf
    combinations= [(gamma, alpha) for gamma in gammas for alpha in alphas]
    for gamma, alpha in tqdm(combinations, desc=kernel_name+" Tuning...", ascii=False, ncols=70):
        # Compute kernel matrix in parallel
        K_train = parallel_kernel_matrix_computation(X_train, kernel_base, verbose, gamma=gamma) 

        if rank==0:
            if verbose: print(f"Rank {rank}: Finished constructing global kernel matrix K_train of shape {K_train.shape}...\n")
        
            # Perform Kernel Ridge Regression on root process
            alpha_hat = kernel_ridge_regression(K_train, y_train, alpha)
            if verbose: print(f"Rank {rank}: Finished computing alpha_hat...\n")

            # Compute test kernel matrix
            K_test = kernel_base(X_test, X_train, gamma=gamma)
            if verbose: print(f"Rank {rank}: Finished calculating global test kernel matrix K_test of shape {K_test.shape} based on X_test of shape {X_test.shape} and X_train of shape {X_train.shape}...\n")

            # Make predictions
            y_pred_train= predict(K_train,alpha_hat)
            y_pred_test = predict(K_test, alpha_hat)
            if verbose: print(f"Rank {rank}: Finished inference to obtain y_pred_train {y_pred_train.shape} y_pred_test {y_pred_test.shape}...\n")
            
            # Evaluate RMSE (Root Mean Squared Error)
            rmse_train= compute_rmse(y_pred_train, y_train)
            rmse_test = compute_rmse(y_pred_test, y_test)
            if verbose: print(f"\nRank {rank} gamma {gamma:.2e} alpha {alpha:.2e}: Finished computing RMSE_train={rmse_train:.4f} RMSE_test={rmse_test:.4f}...\n")

            if rmse_test < best_rmse_test:
                best_gamma = gamma 
                best_alpha = alpha 
                best_rmse_train = rmse_train
                best_rmse_test  = rmse_test 

            if rmse_test < overall_best["best_rmse_test"]: 
                overall_best["best_rmse_train"]  = rmse_train
                overall_best["best_rmse_test"]   = rmse_test
                overall_best["best_kernel"] = kernel_name
                overall_best["best_params"] = f"gamma:{gamma:.2e}, alpha:{alpha:.2e}"

    if rank==0: print(f"\n{kernel_name} ==> Best gamma {best_gamma:.2e}, Best alpha {best_alpha:.2e}, Best RMSE_train {best_rmse_train:.4f} RMSE_test {best_rmse_test:.4f}...\n")
    ###################################################################
    ######### FINISH: Hyper Parameter Tuning for Laplacian Kernel #####
    ###################################################################

    print(f"{'#--------------------------------------------------------------------------------------------------------------' if rank==0 else ''}\n")

    ###################################################################
    ######### START : Hyper Parameter Tuning for Linear Kernel ########
    ######### ------------------------------------------------ ########
    ######### Per 09 June 2024, best_alpha = 1.00e-1           ########
    ######### Per 09 June 2024, best_rmse_train= 70083.2611    ########
    ######### Per 09 June 2024, best_rmse_test = 69158.3004    ########
    ###################################################################
    kernel_name = "Linear Kernel"
    kernel_base = linear_kernel # Hyper parameters: alpha 
    verbose     = False
    alphas      = [1e-6*(10**i) for i in range(1,  9)]  # Alpha is a Regularization parameter for Linear Kernel
    best_alpha  = 1.0
    best_rmse_test   = np.inf
    for alpha in tqdm(alphas, desc=kernel_name+" Tuning...", ascii=False, ncols=70):
        # Compute kernel matrix in parallel
        K_train = parallel_kernel_matrix_computation(X_train, kernel_base, verbose) 

        if rank==0:
            if verbose: print(f"Rank {rank}: Finished constructing global kernel matrix K_train of shape {K_train.shape}...\n")
        
            # Perform Kernel Ridge Regression on root process
            alpha_hat = kernel_ridge_regression(K_train, y_train, alpha)
            if verbose: print(f"Rank {rank}: Finished computing alpha_hat...\n")

            # Compute test kernel matrix
            K_test = kernel_base(X_test, X_train)
            if verbose: print(f"Rank {rank}: Finished calculating global test kernel matrix K_test of shape {K_test.shape} based on X_test of shape {X_test.shape} and X_train of shape {X_train.shape}...\n")

            # Make predictions
            y_pred_train= predict(K_train,alpha_hat)
            y_pred_test = predict(K_test, alpha_hat)
            if verbose: print(f"Rank {rank}: Finished inference to obtain y_pred_train {y_pred_train.shape} y_pred_test {y_pred_test.shape}...\n")
            
            # Evaluate RMSE (Root Mean Squared Error)
            rmse_train= compute_rmse(y_pred_train, y_train)
            rmse_test = compute_rmse(y_pred_test,  y_test)
            if verbose: print(f"\nRank {rank} alpha {alpha:.2e}: Finished computing RMSE_train={rmse_train:.4f} RMSE_test={rmse_test:.4f}...\n")

            if rmse_test < best_rmse_test:
                best_alpha = alpha 
                best_rmse_train = rmse_train
                best_rmse_test  = rmse_test 

            if rmse_test < overall_best["best_rmse_test"]: 
                overall_best["best_rmse_train"]  = rmse_train
                overall_best["best_rmse_test"]   = rmse_test
                overall_best["best_kernel"] = kernel_name
                overall_best["best_params"] = f"alpha:{alpha:.2e}"

    if rank==0: print(f"\n{kernel_name} ==> Best alpha {best_alpha:.2e}, Best RMSE_train {best_rmse_train:.4f}, Best RMSE_test {best_rmse_test:.4f}...\n")
    ###################################################################
    ######### FINISH: Hyper Parameter Tuning for Linear Kernel ########
    ###################################################################

    print(f"{'#--------------------------------------------------------------------------------------------------------------' if rank==0 else ''}\n")

    ###################################################################
    ######### START : Hyper Parameter Tuning for Polynomial Kernel ####
    ######### ---------------------------------------------------- ####
    ######### Per 09 June 2024, best_degree = 3                    ####
    ######### Per 09 June 2024, best_coef0  = 1.00e+0              ####
    ######### Per 09 June 2024, best_alpha  = 1.00e+2              ####
    ######### Per 09 June 2024, best_rmse_train= 60335.5121        ####
    ######### Per 09 June 2024, best_rmse_test = 61791.7949        ####
    ###################################################################
    kernel_name = "Polynomial Kernel"
    kernel_base = polynomial_kernel # Hyper parameters: degree, coef0, alpha
    verbose     = False

    degrees     = [           i for i in range(2,  6)]            
    coef0s      = [       0.2*i for i in range(1,  6)]
    alphas      = [1e-1*(10**i) for i in range(1,  6)]  # Alpha is a Regularization parameter for Polynomial Kernel

    best_degree = 10
    best_coef0  = 1.0
    best_rmse_test   = np.inf
    combinations= [(degree, coef0, alpha) for degree in degrees for coef0 in coef0s for alpha in alphas]
    for degree, coef0, alpha in tqdm(combinations, desc=kernel_name+" Tuning...", ascii=False, ncols=70):
        # Compute kernel matrix in parallel
        K_train = parallel_kernel_matrix_computation(X_train, kernel_base, verbose, degree=degree, coef0=coef0)

        if rank==0:
            if verbose: print(f"Rank {rank}: Finished constructing global kernel matrix K_train of shape {K_train.shape}...\n")
        
            # Perform Kernel Ridge Regression on root process
            alpha_hat = kernel_ridge_regression(K_train, y_train, alpha)
            if verbose: print(f"Rank {rank}: Finished computing alpha_hat...\n")

            # Compute test kernel matrix
            K_test = kernel_base(X_test, X_train, degree=degree, coef0=coef0)
            if verbose: print(f"Rank {rank}: Finished calculating global test kernel matrix K_test of shape {K_test.shape} based on X_test of shape {X_test.shape} and X_train of shape {X_train.shape}...\n")

            # Make predictions
            y_pred_train= predict(K_train,alpha_hat)
            y_pred_test = predict(K_test, alpha_hat)
            if verbose: print(f"Rank {rank}: Finished inference to obtain y_pred_train {y_pred_train.shape} y_pred_test {y_pred_test.shape}...\n")
            
            # Evaluate RMSE (Root Mean Squared Error)
            rmse_train= compute_rmse(y_pred_train, y_train)
            rmse_test = compute_rmse(y_pred_test, y_test)
            if verbose: print(f"\nRank {rank} degree {degree} coef0 {coef0:.2e} alpha {alpha:.2e}: Finished computing RMSE_train={rmse_train:.4f} RMSE_test={rmse_test:.4f}...\n")

            if rmse_test < best_rmse_test:
                best_degree= degree 
                best_coef0 = coef0  
                best_alpha = alpha  
                best_rmse_train = rmse_train
                best_rmse_test  = rmse_test 

            if rmse_test < overall_best["best_rmse_test"]: 
                overall_best["best_rmse_train"]  = rmse_train
                overall_best["best_rmse_test"]   = rmse_test
                overall_best["best_kernel"] = kernel_name
                overall_best["best_params"] = f"degree:{degree}, coef0:{coef0:.2e}, alpha:{alpha:.2e}"

    if rank==0: print(f"\n{kernel_name} ==> Best degree {best_degree}, Best coef0 {best_coef0:.2e}, Best alpha {best_alpha:.2e}, Best RMSE_train {best_rmse_train:.4f}, Best RMSE_test {best_rmse_test:.4f}...\n")
    ###################################################################
    ######### FINISH: Hyper Parameter Tuning for Polynomial Kernel ####
    ###################################################################

    print(f"{'#--------------------------------------------------------------------------------------------------------------' if rank==0 else ''}\n")

    ###################################################################
    ######### START : Hyper Parameter Tuning for Sigmoid Kernel #######
    ######### ---------------------------------------------------- ####
    ######### Per 09 June 2024, best_gamma  = 1.00e-3              ####
    ######### Per 09 June 2024, best_coef0  = 8.00e-1              ####
    ######### Per 09 June 2024, best_alpha  = 1.00e-5              ####
    ######### Per 09 June 2024, best_rmse_train= 62942.9750        ####
    ######### Per 09 June 2024, best_rmse_test = 63110.9491        ####
    ###################################################################
    kernel_name = "Sigmoid Kernel"
    kernel_base = sigmoid_kernel # Hyper parameters: gamma, coef0, alpha
    verbose     = False

    gammas      = [1e-6*(10**i) for i in range(1,  6)]  # Gamma is a Kernel coefficient for Sigmoid Kernel. 
    coef0s      = [       0.2*i for i in range(1,  5)]
    alphas      = [1e-8*(10**i) for i in range(1,  6)]  # Alpha is a Regularization parameter for Polynomial Kernel

    best_gamma  = 1.0
    best_coef0  = 1.0
    best_rmse_test   = np.inf
    combinations= [(gamma, coef0, alpha) for gamma in gammas for coef0 in coef0s for alpha in alphas]
    for gamma, coef0, alpha in tqdm(combinations, desc=kernel_name+" Tuning...", ascii=False, ncols=70):
        # Compute kernel matrix in parallel
        K_train = parallel_kernel_matrix_computation(X_train, kernel_base, verbose, gamma=gamma, coef0=coef0)

        if rank==0:
            if verbose: print(f"Rank {rank}: Finished constructing global kernel matrix K_train of shape {K_train.shape}...\n")
        
            # Perform Kernel Ridge Regression on root process
            alpha_hat = kernel_ridge_regression(K_train, y_train, alpha)
            if verbose: print(f"Rank {rank}: Finished computing alpha_hat...\n")

            # Compute test kernel matrix
            K_test = kernel_base(X_test, X_train, gamma=gamma, coef0=coef0)
            if verbose: print(f"Rank {rank}: Finished calculating global test kernel matrix K_test of shape {K_test.shape} based on X_test of shape {X_test.shape} and X_train of shape {X_train.shape}...\n")

            # Make predictions
            y_pred_train= predict(K_train,alpha_hat)
            y_pred_test = predict(K_test, alpha_hat)
            if verbose: print(f"Rank {rank}: Finished inference to obtain y_pred_train {y_pred_train.shape} y_pred_test {y_pred_test.shape}...\n")
            
            # Evaluate RMSE (Root Mean Squared Error)
            rmse_train= compute_rmse(y_pred_train, y_train)
            rmse_test = compute_rmse(y_pred_test, y_test)
            if verbose: print(f"\nRank {rank} gamma {gamma:.2e} coef0 {coef0:.2e} alpha {alpha:.2e}: Finished computing RMSE_train={rmse_train:.4f} RMSE_test={rmse_test:.4f}...\n")

            if rmse_test < best_rmse_test:
                best_gamma = gamma 
                best_coef0 = coef0  
                best_alpha = alpha  
                best_rmse_train = rmse_train
                best_rmse_test  = rmse_test 

            if rmse_test < overall_best["best_rmse_test"]: 
                overall_best["best_rmse_train"]  = rmse_train
                overall_best["best_rmse_test"]   = rmse_test
                overall_best["best_kernel"] = kernel_name
                overall_best["best_params"] = f"gamma:{gamma:.2e}, coef0:{coef0:.2e}, alpha:{alpha:.2e}"

    if rank==0: print(f"\n{kernel_name} ==> Best gamma {best_gamma:.2e}, Best coef0 {best_coef0:.2e}, Best alpha {best_alpha:.2e}, Best RMSE_train {best_rmse_train:.4f}, Best RMSE_test {best_rmse_test:.4f}...\n")
    ###################################################################
    ######### FINISH: Hyper Parameter Tuning for Sigmoid Kernel #######
    ###################################################################

    print(f"{'#--------------------------------------------------------------------------------------------------------------' if rank==0 else ''}\n")

    return overall_best



if __name__ == "__main__":
    start = time()

    rank = MPI.COMM_WORLD.Get_rank()  # Get rank of current process
    overall_best = main()
    if rank==0: print(f"Overall Best: base kernel name={overall_best['best_kernel']}, RMSE_train={overall_best['best_rmse_train']}, RMSE_test={overall_best['best_rmse_test']}, best hyperparameters={overall_best['best_params']}")
    MPI.Finalize()
    
    end = time()
    print(f"Rank {rank}: Execution Time: {strftime('%H:%M:%S', gmtime(end-start))}\n")
