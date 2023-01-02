from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.models import Model, Sequential
import pandas as pd

def create_encoder(input_dim ,model ,encoded_layer):
    input_used= Input(shape=(input_dim,))
    encoded = model.layers[0](input_used)
    for i in range(1,encoded_layer):
        encoded = model.layers[i](encoded)

    return Model(input_used, encoded)

def AE(training_set, validation_set, epochs, batch_size, optimizer, loss, dimensions, activations, encoded_layer, alphas = None, dropouts = None, verbose=1):
    """  """

    if alphas is None:
        alphas = [0.0] * (len(dimensions) - 2)
    if dropouts is None:
        dropouts = [0.0] * (len(dimensions) - 2)
    
    if (len(dimensions) - 1) != len(activations):
        raise ValueError('There must be as many activation functions as there are layer dimensions minus 1')
    
    if len(alphas) != (len(dimensions) - 2) :
        raise ValueError("There must be as many alpha values as there are layer dimensions minus 2 (here {0})".format(len(dimensions)-2))
    
    if len(dropouts) != (len(dimensions) - 2):
        raise ValueError("There must be as many dropout values as there are layer dimensions minus 2 (here {0})".format(len(dimensions)-2))
    
    if len(dimensions) < 3:
        raise ValueError("One needs to specify at least 3 layers")
    
    if not all(isinstance(x, int) for x in dimensions) :
        raise ValueError("The dimensions of a layer needs to be an integer")
    
    accepted_functions = ['relu','sigmoid','softmax','softplus', 'softsign','tanh','selu', 'elu','exponential', 'linear']
    if any([v not in accepted_functions for v in activations]) :
        raise ValueError("The activation function should be one of those : 'relu','sigmoid','softmax','softplus', 'softsign','tanh','selu', 'elu','exponential','linear'")
    
    if dimensions[-1] != dimensions[0]:
        raise ValueError('The dimension of the first layer should be equal to the dimension of the latest')
    
    if not all(isinstance(x, float) for x in alphas) :
        raise ValueError("All values of alpha need to be a float")   
        
    if not all(isinstance(x, float) for x in dropouts) or all((x<0)|(x>1) for x in dropouts)  :
        raise ValueError("All values of dropout need to be a float") 
    
    if  any((x<0)|(x>1) for x in dropouts)  :
        raise ValueError("All values of dropout should be between 0 and 1")
    
    if  any((x<0)|(x>1) for x in alphas)  :
        raise ValueError("All values of alpha should be between 0 and 1")
    
    
    # Initialisation of the model
    model=Sequential()
    
    # First layer
    model.add(Dense(dimensions[1], input_dim= dimensions[0], activation=activations[0]))
    model.add(LeakyReLU(alpha=alphas[0]))
    model.add(Dropout(rate = dropouts[0]))
    
    # Middle layers
    for dim in range(1,len(dimensions) - 2):
        model.add(Dense(dimensions[dim+1], activation=activations[dim]))
        model.add(LeakyReLU(alpha=alphas[dim]))
        model.add(Dropout(rate = dropouts[dim]))
        
    # Last layer
    model.add(Dense(dimensions[-1], activation= activations[-1]))
    
    model.compile(optimizer=optimizer, loss=loss)


    # Training
    autoencoder = model.fit(training_set, training_set,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(validation_set, validation_set),
                            verbose=verbose)

    encoder = create_encoder(dimensions[0] , model ,encoded_layer)

    loss = autoencoder.history['loss']
    val_loss = autoencoder.history['val_loss']
    losses = pd.DataFrame({'Train_loss' : loss, 'Validation_loss' : val_loss})


    return losses, encoder, autoencoder, model
