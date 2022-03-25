# quick script used to calculate the output size required based on the layers used

def calc_layer(w,k,s,p):
    # w = input size
    # k = filter size
    # s = stride
    # p = padding

    return (((w - k + 2*p)/s)+1)
    

#based on the current layout used in the code 
size = 0

# Conv2D layer
size = size + calc_layer(0,)







