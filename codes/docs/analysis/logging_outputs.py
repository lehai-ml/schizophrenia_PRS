import numpy as np

def logging_output_as_array(func,argument):
    def wrapper(*args,**kwargs):
        output=func(*args,**kwargs)
        np.asarray(output)
        with open(argument,'ab') as f:
            np.savetxt(f,np.asarray([output]),delimiter='\t')
    return wrapper