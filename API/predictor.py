import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import ImageOps,Image
import cv2


def predict_with_model(model,img,target_size=(256,256),class_names=['Cas','Cos','Gum','MC','OC','OLP','OT']):
    
  
    

    # # Check if file exists
    # if not os.path.isfile(img_path):
    #     raise FileNotFoundError(f"No such file: '{img_path}'")
    
      # Resize the image to the target size
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the input shape expected by the model (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image data (as done during training, e.g., rescaling)
    img_array = img_array / 255.0

    predictions = model.predict(img_array) 
     # Get the predicted class index
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Get the confidence of the prediction
    confidence = np.max(predictions, axis=1)[0]
    if class_names:
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = str(predicted_class_idx)
  

    return predicted_class,confidence 




# if __name__ =="__main__":
#     image_path= "D:\Projects Computer Vision\Cv internship Cellula Project 7\Data\Teeth DataSet\Teeth_Dataset\Validation\MC\mc_1201_0_1698.jpg"

#     model=tf.keras.models.load_model('model.keras')
#     predictions, confidence = predict_with_model( model,  image_path)


#     print("prediction =",predictions)
#     print(confidence)