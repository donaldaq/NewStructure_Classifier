'''
 make weight for balanced each classes 

'''
# parameter: imagedataset image, length of image dataset classes  
def make_weights_for_balanced_classes(images, nclasses):
    """ make weight balanced classes

    Args:
        images (Object): Each of images
        nclasses (Integer): number of classes

    Returns:
        [type]: [description]
    """

    print("balanced classes set")
    count = [0] * nclasses
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))
    for i in range(nclasses):
        #print(i)
        #print(float(count[i]))
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

