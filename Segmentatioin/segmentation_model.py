import tensorflow
import segmentation_models as sm
sm.set_framework('tf.keras')

BACKBONE = 'inceptionv3'
activation='softmax'
encoder_freeze = False
num_classes = 3

preprocess_input = sm.get_preprocessing(BACKBONE)
def model(dropout_value):
    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, 
                    input_shape = (256,256,3), 
                    encoder_weights='imagenet', 
                    classes=num_classes, 
                    activation=activation,
                    encoder_freeze = encoder_freeze)    
    model.summary()
    model_input = model.input
    model_output = model.get_layer('final_conv').output    
    model_output = tensorflow.keras.layers.Dropout(dropout_value)(model_output)    
    output = tensorflow.keras.layers.Activation(activation, name=activation)(model_output)
    model_dp = tensorflow.keras.models.Model(model_input, output)     
    return model_dp, preprocess_input   
