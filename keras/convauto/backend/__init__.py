from keras import backend as K

if K._BACKEND == 'theano':
    from .theano_backend import *
elif K._BACKEND == 'tensorflow':
    from .tensorflow_backend import *
else:
    raise Exception('Unknown backend: ' + str(K._BACKEND))