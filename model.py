from keras.models import model_from_json
import numpy as np

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self,model_json,model_weights):
        with open(model_json,'r') as json_file:
            loaded_model=json_file.read()
            self.loaded_model=model_from_json(loaded_model)
        self.loaded_model.load_weights(model_weights)

    def predict_emotion(self,img):
        self.preds=self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]