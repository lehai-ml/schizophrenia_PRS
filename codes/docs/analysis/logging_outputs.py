import numpy as np

def logging_output_as_array(func,argument):
    """
    Decorator to save np.arrays into txt files.
    Arguments: 
        func: original functions
        argument: the location of the file
    Returns
        outputs to txt file.
    Note:
        np.loadtxt(/path/to/file): to retrive the np array.    
    
    """
    def wrapper(*args,**kwargs):
        output=func(*args,**kwargs)
        np.asarray(output)
        with open(argument,'ab') as f:
            np.savetxt(f,np.asarray([output]),delimiter='\t')
    return wrapper