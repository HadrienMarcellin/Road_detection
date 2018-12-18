import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from BBQPL_image_processing import contineous_to_binary_mask

class Metrics(Callback):
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.model.save('model_epoch_{0}'.format(epoch))
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = contineous_to_binary_mask(val_predict).ravel()
        val_targ = self.validation_data[1].ravel()
        _val_f1 = f1_score(val_targ.ravel(), val_predict, average='binary', pos_label = 1, labels=[0, 1])
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: {0:.5f} — val_precision: {1:.5f} — val_recall {2:.5f}".format(_val_f1, _val_precision, _val_recall))
        return

