from keras import backend as K
import backend as convauto_backend
from keras.layers.convolutional import UpSampling2D, Convolution2D

'''TODO: now that we know how to get the batch size at exec time, we can finally
use maxpool_with_argmax'''

class UnPool2D(UpSampling2D):
    '''
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.

    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
    '''
    def __init__(self, pool2d, *args, **kwargs):
        self._pool2d = pool2d
        self.size = pool2d.pool_size
        super(UnPool2D, self).__init__(*args, **kwargs)

    def call(self, X, mask=None):
        if self.dim_ordering == 'th':
            output = K.repeat_elements(X, self.size[0], axis=2)
            output = K.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = K.repeat_elements(X, self.size[0], axis=1)
            output = K.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        return convauto_backend.unpool2d(self._pool2d.output, self._pool2d.input, output)

class ConvolutionalTranspose2D(Convolution2D):
    '''
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, nb_row, nb_col)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, nb_row, nb_col, nb_filter)` if dim_ordering='tf'.
    '''
    def __init__(self, conv_layer, *args, **kwargs):
        self._conv_layer = conv_layer

        kwargs['nb_filter'] = self._conv_layer.input_shape[1]
        #kwargs['nb_filter'] = self._conv_layer.nb_filter
        kwargs['nb_row'] = self._conv_layer.nb_row
        kwargs['nb_col'] = self._conv_layer.nb_col
        kwargs['border_mode'] = self._conv_layer.border_mode
        super(ConvolutionalTranspose2D, self).__init__(*args, **kwargs)

        #self.nb_out_channels = self._conv_layer.input_shape[1]
        if self.dim_ordering == 'th':
            self.nb_out_channels = self._conv_layer.input_shape[1]
        elif self.dim_ordering == 'tf':
            self.nb_out_channels = self._conv_layer.input_shape[0]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        subsample = self._conv_layer.subsample
        output_shape = convauto_backend.get_shape_with_batch(self._conv_layer.input)
        border_mode = self._conv_layer.border_mode

        output = convauto_backend.conv2d_transpose(x, self.W, output_shape,
                                                   subsample, border_mode,
                                                   dim_ordering=self.dim_ordering)

        if self.bias:
            if self.dim_ordering == 'th':
                output += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        output = self.activation(output)
        return output

    def build(self,input_shape):
        self.W = self._conv_layer.W
        '''
        if self.dim_ordering == 'th':
            self.W_shape = (self.nb_out_channels, self.nb_filter, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            raise NotImplementedError()
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        '''
        self.b = K.zeros((self.nb_out_channels,))
        self.params = [self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        output_shape = list(super(ConvolutionalTranspose2D, self).output_shape)

        if self.dim_ordering == 'th':
            output_shape[1] = self.nb_out_channels
        elif self.dim_ordering == 'tf':
            output_shape[0] = self.nb_out_channels
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        return tuple(output_shape)

