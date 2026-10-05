import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import json, yaml
import utils
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss) 


def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


class Multifold():
    def __init__(self,version,config_file='config_omnifold.json',verbose=False,plot_folder="../plots/",weights_folder="../weights/"):
        self.opt = LoadJson(config_file)
        self.plot_folder = plot_folder
        self.niter = self.opt['NITER']
        self.ntrial = self.opt['NTRIAL']
        self.version=version
        self.mc_gen = None
        self.mc_reco = None
        self.data=None
        self.verbose=verbose
        self.model1 = []
        self.model2 = []
                
        self.weights_folder = weights_folder
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
    def Unfold(self, start_iter=0):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']

        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])
            
        for i in range(start_iter, self.niter):
            print("ITERATION: {}".format(i + 1))
            thisLR = float(self.opt['LR'])
            # (Optional) Reduce initial learning rate for step 1 if this is not the first iteration
            if i > 0:
                thisLR *= 0.5
            for j in range(self.ntrial):
                self.CompileModel(thisLR,self.model1[j])
                self.RunStep1(i, j)
            new_weights=self.reweightAverage(self.mc_reco,self.model1)
            
            self.weights_pull = self.weights_push
            self.weights_pull[self.pass_gen] *= new_weights
                    
            for j in range(self.ntrial):
                self.CompileModel(float(self.opt['LR']),self.model2[j])
                self.RunStep2(i, j)
            new_weights=self.reweightAverage(self.mc_gen,self.model2)
            self.weights_push = new_weights

            np.save(os.path.join(self.weights_folder, 'Step1_Iter{}_{}_PullWeights.npy'.format(i, self.opt['NAME'])), self.weights_pull)
            np.save(os.path.join(self.weights_folder, 'Step2_Iter{}_{}_PushWeights.npy'.format(i, self.opt['NAME'])), self.weights_push)
                
    def RunStep1(self,i,trial):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")
        self.RunModel(
            np.concatenate([self.mc_reco, self.data]),
            np.concatenate([self.labels_mc, self.labels_data]),
            np.concatenate([self.weights_push[self.pass_gen]*self.weights_mc_reco,self.weights_data]),
            trial,i,self.model1[trial],stepn=1,
        )
        
        

    def RunStep2(self,i, trial):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")

        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((self.labels_gen_prior, self.labels_gen)),
            np.concatenate((self.weights_mc, self.weights_mc*self.weights_pull)),
            trial,i,self.model2[trial],stepn=2,
        )


    def RunModel(self,sample,labels,weights,trial,iteration,model,stepn):    
        verbose = 1
        
        callbacks = [
            ReduceLROnPlateau(patience=int(0.8*self.opt['NPATIENCE']), min_lr=1e-6, factor=0.3, verbose=verbose),
            EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True)
        ]

        # (Optional) Can reset model weights to a pre-set point on each new iteration, instead of carrying over from previous iteration
        # if stepn == 1:
        #     model.load_weights("step1_model.h5")
        # if stepn == 2:
        #     model.load_weights("step2_model.h5")
        
        callbacks.append(
            ModelCheckpoint('{}/{}_trial{}_iter{}_step{}.h5'.format(
                self.weights_folder,self.version,trial,iteration,stepn),
                            save_best_only=True,mode='auto',period=1,save_weights_only=False))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(sample, labels, weights,test_size=0.2)
        history = model.fit(X_train_1,np.stack((Y_train_1, w_train_1), axis=1),
                          epochs=self.EPOCHS,
#                          steps_per_epoch=int(X_train_1.shape[0]/self.BATCH_SIZE)-1,
                          validation_data=(X_test_1,np.stack((Y_test_1, w_test_1), axis=1)),
#                          validation_steps=int(X_test_1.shape[0]/self.BATCH_SIZE)-1,
                          callbacks=callbacks,
                          verbose=verbose,
                          batch_size=self.BATCH_SIZE)
        json.dump(str(history.history), open('{}/{}_iter{}_step{}_History.json'.format(self.weights_folder,self.version,iteration,stepn), 'w'))

    def Preprocessing(self,weights_mc_reco=None,weights_mc=None,weights_data=None,pass_reco=None,pass_gen=None,start_iter=0):
        self.PrepareWeights(weights_mc_reco,weights_mc,weights_data,pass_reco,pass_gen)
        self.PrepareInputs()
        self.PrepareModel(nvars1=self.mc_reco.shape[1], nvars2 = self.mc_gen.shape[1],start_iter=start_iter)

    def PrepareWeights(self,weights_mc_reco,weights_mc,weights_data,pass_reco,pass_gen):
        self.pass_gen = np.copy(pass_gen).astype('bool')
        self.pass_reco = np.copy(pass_reco).astype('bool')

        if weights_mc_reco is None:
            if self.verbose: print("No MC reco weights provided, making one filled with 1s")
            self.weights_mc_reco = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc_reco = np.copy(weights_mc_reco)
        
        if weights_mc is None:
            if self.verbose: print("No MC weights provided, making one filled with 1s")
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = np.copy(weights_mc)

        if weights_data is None:
            if self.verbose: print("No data weights provided, making one filled with 1s")
            self.weights_data = np.ones(self.data.shape[0])
        else:
            self.weights_data = np.copy(weights_data)
            
    def CompileModel(self,lr,model):
        opt = tensorflow.keras.optimizers.Adam(learning_rate=float(self.opt['LR']))

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer=opt,experimental_run_tf_function=False)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))
        self.labels_data = np.ones(len(self.data))
        self.labels_gen_prior = np.zeros(len(self.mc_gen))
        self.labels_gen = np.ones(len(self.mc_gen))

    def PrepareModel(self,nvars1,nvars2,start_iter=0):
        for i in range(self.ntrial):
            inputs1,outputs1 = MLP_Step1(nvars1)
            inputs2,outputs2 = MLP_Step2(nvars2)
            self.model1.append(Model(inputs=inputs1, outputs=outputs1))
            self.model2.append(Model(inputs=inputs2, outputs=outputs2))

            # Can load pre-trained weights here
            # self.model1[-1].load_weights("Step1_Pretrain.h5")
            # self.model2[-1].load_weights("Step2_Pretrain.h5")

    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=10000)[:,:1],posinf=1,neginf=0)
        weights = f / (1. - f)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

    def reweightAverage(self,events,models):
        f = np.nan_to_num(models[0].predict(events, batch_size=10000)[:,:1],posinf=1,neginf=0)[:,0]
        for i in range(1, self.ntrial):
            f = f + np.nan_to_num(models[i].predict(events, batch_size=10000)[:,:1],posinf=1,neginf=0)[:,0]
        f = f / self.ntrial
        weights = f / (1. - f)
        return np.squeeze(np.nan_to_num(weights,posinf=1))

    def LoadModel(self,iteration):
        model_name = '{}/{}_iter{}_step2.h5'.format(
            self.weights_folder,self.version,iteration)
        self.model2.load_weights(model_name)

# Can use different network configurations for step 1 and step 2
def MLP_Step1(nvars):
    ''' Define a simple fully conneted model to be used during unfolding'''
    inputs = Input((nvars, ))
    layer = Dense(100,activation='relu')(inputs)
    layer = Dense(100,activation='relu')(layer)
    layer = Dense(100,activation='relu')(layer)
    outputs = Dense(1,activation='sigmoid')(layer)
    return inputs,outputs

def MLP_Step2(nvars):
    ''' Define a simple fully conneted model to be used during unfolding'''
    inputs = Input((nvars, ))
    layer = Dense(100,activation='relu')(inputs)
    layer = Dense(100,activation='relu')(layer)
    layer = Dense(100,activation='relu')(layer)
    outputs = Dense(1,activation='sigmoid')(layer)
    return inputs,outputs

