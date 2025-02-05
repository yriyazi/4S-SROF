#importing libraries
import  numpy                                   as      np
import  tensorflow                              as      tf
from    tensorflow                              import  keras
from    tensorflow.keras                        import  layers       
from    tensorflow.keras.preprocessing.image    import  img_to_array

# Define the DepthToSpace layer
class DepthToSpace(layers.Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(DepthToSpace, self).__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.upscale_factor)

# Build the model architecture and load the trained weights
def model_architecture(weight_address):
    # Initial values
    upscale_factor = 3
    channels = 1
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    # Model structure
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = DepthToSpace(upscale_factor=upscale_factor)(x)
    
    # Load model
    model = keras.Model(inputs, outputs)
    model.load_weights(weight_address)
    
    return model

# Optimize model prediction with @tf.function
@tf.function
def predict(input_tensor, model):
    input_tensor_tf = tf.convert_to_tensor(input_tensor)
    out = model(input_tensor, training=False)
    out_img_y = out[0, :, :, 0].numpy()  # Remove batch and channel dimensions
    out_img_y = (out_img_y * 255.0).clip(0, 255).astype("uint8")

    return out_img_y


if __name__ +"__main__":
    model = model_architecture("SuperRes_weights.h5")



