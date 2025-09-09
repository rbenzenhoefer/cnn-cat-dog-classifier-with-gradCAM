'''Imports'''
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.models import load_model

'''Loading the model and the test image'''
model = load_model(os.path.join('models', 'catdogmodel.h5'))
image_path = os.path.join('gradcamtestdog.jpg')
img = cv2.imread(image_path)    
img_resized = cv2.resize(img, (256, 256))
img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
'''Prediction Logic'''
prediction = model.predict(img_array, verbose=0)[0][0]
animal = "Dog" if prediction > 0.5 else "Cat"
confidence = prediction if prediction > 0.5 else 1 - prediction
print(f"Prediction: {animal}")
print(f"Confidence: {confidence:.3f}")
print(f"Raw Output: {prediction:.3f}")


'''Creating the gradCAM model'''
def create_gradcam_heatmap(model, img_array, target_layer='conv2d_1'):
    grad_model = tf.keras.Model(
        inputs=model.inputs,  
        outputs=[model.get_layer(target_layer).output, model.outputs[0]]  
    )
    
    '''Compute gradients'''
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

'''GradCAM analysis for the test image'''
def analyze_image_with_gradcam(model, image_path, target_layer='conv2d_1', alpha=0.6):
  
    '''Creating the Heatmap'''
    heatmap = create_gradcam_heatmap(model, img_array, target_layer)
    heatmap_resized = cv2.resize(heatmap, (256, 256))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        
    '''Normalizing the image and creating an overlay'''
    img_normalized = img_resized / 255.0
    superimposed = alpha * heatmap_colored + (1 - alpha) * img_normalized
        
    '''Visualization'''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
    axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Image\n{os.path.basename(image_path)}')
    axes[0].axis('off')
        
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title(f'Grad-CAM Heatmap\nLayer: {target_layer}')
    axes[1].axis('off')
        
    axes[2].imshow(superimposed)
    axes[2].set_title(f'Prediction: {animal}\nConfidence: {confidence:.2f}')
    axes[2].axis('off')
        
    plt.suptitle('Grad-CAM Analysis - Cat vs Dog Classifier', fontsize=16)
    plt.tight_layout()
    plt.show()
        
    return heatmap, superimposed


create_gradcam_heatmap(model, img_array, target_layer='conv2d_1')
analyze_image_with_gradcam(model, image_path, target_layer='conv2d_1', alpha=0.6)