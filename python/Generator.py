import numpy as np
import uproot
import scipy.stats as stats
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.preprocessing import MinMaxScaler


def main():
    '''
    Generate fake datasets for reweighting paper.
    Four datasets are generated, two 2D and two 4D.
    For each of the 2D and 4D, there is a "target" dataset, which the 
    reweighting aims to reconstruct, and a "source" dataset, which is 
    reweighted in order reconstruct the target.
    NOTE: Removed dependency on ROOT, so cannot write to file. Will 
          replace with uproot, if needed (and possible).
    '''
    source_2D = makeDataset(2, 1e6, random_states=[74205,75633])
    scaler = MinMaxScaler(copy=False)
    scaler.fit(source_2D)
    scaler.transform(source_2D)
    target_2D = makeDataset(2, 1e6, random_states=[556354,638901])
    scaler.transform(target_2D)
    target_2D = target_2D[
                            eff2DFunc([target_2D[:,0], target_2D[:,1]]) 
                            > np.random.rand(len(target_2D))
                         ]
 #   writeRootFile(source_2D, target_2D, "test.root")

    source_4D = makeDataset(4, 1e6, random_states=[34154,27164992])
    scaler_4D = MinMaxScaler(copy=False)
    scaler_4D.fit(source_4D)
    scaler_4D.transform(source_4D)
    target_4D = makeDataset(4, 1e6, random_states=[1632496,5551571])
    scaler_4D.transform(target_4D)
    target_4D = target_4D[
                            eff4DFunc([
                                        target_4D[:,0], 
                                        target_4D[:,1],
                                        target_4D[:,2], 
                                        target_4D[:,3],
                            ]) 
                            > np.random.rand(len(target_4D))
                         ]
#    writeRootFile(source_4D, target_4D, "test.root", mode="update")

def makeDataset(n_dimensions, n_total, random_states=[None, None]):
    '''
    Generate a n_dimensions-D dataset by sampling from two Gaussian of fixed properties.
    Inputs: n_dimension   = number of dimensions 
            n_total       = total number of events to generate
            random_states = list of two numpy.random.RandomState objects, or 
                            integer to seed internal RandomState objects
    Output: array containing generated n_dimensional-D data
    '''
    # Create the covariance matrices for the two component Gaussians
    # random_states are specified for reproducibility
    cov1 = make_sparse_spd_matrix(
                                    dim=n_dimensions, 
                                    alpha=0.1, 
                                    random_state=47, 
                                    norm_diag=True,
                                 )
    cov2 = make_sparse_spd_matrix(
                                    dim=n_dimensions, 
                                    alpha=-0.5, 
                                    random_state=1701, 
                                    norm_diag=True,
                                 )
    # Create mean position of first Gaussian.
    np.random.seed(52)
    mu1 = np.random.rand(1,n_dimensions)[0]

    # Create data from first Gaussian component
    X1 = stats.multivariate_normal.rvs(
                                        mean=mu1, 
                                        cov=cov1, 
                                        size=int(0.667*n_total), 
                                        random_state=random_states[0],
                                      )
    # Second Gaussian mean is fixed to be shifted by -1 from that of first
    X2 = stats.multivariate_normal.rvs(
                                        mean=mu1-1., 
                                        cov=cov2, 
                                        size=int(0.333*n_total), 
                                        random_state=random_states[1]   
                                      )
    return np.append(X1, X2, axis=0)


def eff2DFunc(X):
    ''' 
        Function that defines the efficiency as a function of x_i, where i in {1,2}
        Inputs: X, numpy array containing the two dimensions
        Outputs: efficiency
    '''
    a0 = 0.5; a1 = 3*0.158;
    b2 = -1
    c2 = 0
#    return a0*X[0]+a1*np.exp(b2*X[1]+c2)
    #Function from KG
    return 0.72*(0.5*np.exp(-((np.sqrt(X[0]*X[1]+0.01)-0.4)/0.15)**2)+0.75*np.exp(-((X[1]-0.3)/0.25)**2))



def eff4DFunc(X):
    ''' 
        Function that defines the efficiency as a function of x_i, where i in {1,2,3,4}
        Inputs: X, numpy array containing the four dimensions
        Outputs: efficiency
    '''
    a0 = 0.125; a1 = 0.25; a2 = -0.125; a3 = 0.25/np.pi;
    b2 = -1; b4 = 0.5*np.pi;
    c2 = 0; c4 = 0.5*np.pi;
    return a0+a1*X[1]+a2*np.exp(b2*X[2]+c2)+a3*np.sin(b4*X[3]+c4)

#def writeRootFile(source, target, filename, mode="recreate"):
#    '''
#    Writes the source and target datasets to a root file.
#    Inputs: source and target = np.arrays to write to file.
#            filename = name of the root file to write to.
#            mode = 'recreate' will overwrite an existing file named 'filename'
#                   'update' will write trees into existing file 'filename'
#    Output: None
#    Method: the input numpy ndarrays are converted to recarrays, then 
#            root_numpy is used to write these to a root file.
#    '''
#    # Work out names for branches and add formats
#    nD = source.shape[1]
#    branchnames = [] 
#    for i in range(nD):
#       branchnames.append("X{}".format(i+1)) 
#    branchnames = ",".join(branchnames) 
#    formats = ",".join(["f8"]*nD)
#    print("Writing {} source events, with branches {} in formats {}".
#           format(len(source), branchnames, formats)
#         )
#    
#    source_root = np.core.records.fromarrays(
#        source.transpose(), names=branchnames, formats=formats,
#    )
#    root_numpy.array2root(
#        source_root, filename, treename="source_{}D".format(nD), mode=mode
#    )
#    target_root = np.core.records.fromarrays(
#        target.transpose(), names=branchnames, formats=formats,
#    )
#    root_numpy.array2root(
#        target_root, filename, treename="target_{}D".format(nD), mode="update"
#    )

if __name__ == "__main__":
    main()
