from typing import Callable
from NetworkModelBase import NetworkModelBase
import tensorflow as tf
import numpy as np

class Conv_BN_Act(tf.keras.layers.Layer):
    def __init__(self, filters, KernelSize, act_type, is_bn=True, padding='same', strides=1, conv_tran=False):
        super(Conv_BN_Act, self).__init__()
        if conv_tran:
            self.conv = tf.keras.layers.Conv2DTranspose(filters,
                                               KernelSize,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=False)
        else:
            self.conv = tf.keras.layers.Conv2D(filters,
                                      KernelSize,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=False)

        self.is_bn = is_bn
        if is_bn:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9)

        if act_type == 'LeakyReLU':
            self.act = tf.keras.layers.LeakyReLU(alpha=0.2)
            self.erase_act = False
        elif act_type == 'ReLU':
            self.act = tf.keras.layers.ReLU()
            self.erase_act = False
        elif act_type == 'Tanh':
            self.act = tf.keras.layers.Activation(tf.tanh)
            self.erase_act = False
        elif act_type == '':
            self.erase_act = True
        else:
            raise ValueError

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.is_bn else x
        x = x if self.erase_act else self.act(x)
        return x

class Encoder(tf.keras.layers.Layer):
    """ DCGAN ENCODER NETWORK
    """
    def __init__(self, intImageSize, intLatentDim, intInputDim, intFilter):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ndf(int): num of discriminator(Encoder) filters
        """
        super(Encoder, self).__init__()
        assert intImageSize % 16 == 0, "isize has to be a multiple of 16"

        self.in_block = Conv_BN_Act(filters=intFilter,
                                    KernelSize=4,
                                    activation='LeakyReLU',
                                    is_bn=False,
                                    strides=2)
        csize, cndf = intImageSize / 2, intFilter

        self.body_blocks = []
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            body = Conv_BN_Act(filters=out_feat,
                               KernelSize=4,
                               activation='LeakyReLU',
                               strides=2)
            self.body_blocks.append(body)
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        self.out_conv = tf.keras.layers.Conv2D(filters=intLatentDim,
                                      KernelSize=4,
                                      padding='valid')

    def call(self, x):
        x = self.in_block(x)
        for block in self.extra_blocks:
            x = block(x)
        for block in self.body_blocks:
            x = block(x)
        last_features = x
        out = self.out_conv(last_features)

        return out

class Decoder(tf.keras.layers.Layer):
    def __init__(self, intImageSize, intLatentDim, intInputDim, intFilter):
        """
        Params:
            isize(int): input image size
            nz(int): num of latent dims
            nc(int): num of input dims
            ngf(int): num of Generator(Decoder) filters
        """
        super(Decoder, self).__init__()
        assert intImageSize % 16 == 0, "isize has to be a multiple of 16"
        cngf, tisize = intFilter // 2, 4
        while tisize != intImageSize:
            cngf = cngf * 2
            tisize = tisize * 2

        self.in_block = Conv_BN_Act(filters=cngf,
                                    ks=4,
                                    act_type='ReLU',
                                    padding='valid',
                                    conv_tran=True)

        csize, _ = 4, cngf
        self.body_blocks = []
        while csize < intImageSize // 2:
            body = Conv_BN_Act(filters=cngf // 2,
                               ks=4,
                               act_type='ReLU',
                               strides=2,
                               conv_tran=True)
            self.body_blocks.append(body)
            cngf = cngf // 2
            csize = csize * 2

        self.out_block = Conv_BN_Act(filters=intInputDim,
                                     ks=4,
                                     act_type='Tanh',
                                     strides=2,
                                     is_bn=False,
                                     conv_tran=True)

    def call(self, x):
        x = self.in_block(x)
        for block in self.body_blocks:
            x = block(x)
        for block in self.extra_blocks:
            x = block(x)
        x = self.out_block(x)
        return x

class NetG(tf.keras.Model):
    def __init__(self, imageSize, intLatentDim, intChannels, intDiscriminatorFilter):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(imageSize, intLatentDim, intChannels, intDiscriminatorFilter)
        self.decoder = Decoder(imageSize, intLatentDim, intChannels, intDiscriminatorFilter)
        self.encoder2 = Encoder(imageSize, intLatentDim, intChannels, intDiscriminatorFilter)

    def call(self, x):
        latent_i = self.encoder1(x)
        gen_img = self.decoder(latent_i)
        latent_o = self.encoder2(gen_img)
        return latent_i, gen_img, latent_o

    def num_params(self):
        return sum(
            [np.prod(var.shape.as_list()) for var in self.trainable_variables])

class NetD(tf.keras.Model):
    def __init__(self, imageSize, intLatentDim, intChannels, intDiscriminatorFilter):
        super(NetD, self).__init__()

        self.encoder = Encoder(imageSize, intLatentDim, intChannels, intDiscriminatorFilter)

        self.sigmoid = tf.keras.layers.Activation(tf.sigmoid)

    def call(self, x):
        output, last_features = self.encoder(x)
        output = self.sigmoid(output)
        return output, last_features

class ModelTest(NetworkModelBase) :
    intDiscriminatorFilter = 64
    intGeneratorFilter = 64
    intLatentDim = 100
    fGloss = 0
    fDloss = 0
    def __init__(self, **kwargs) :
        while True :
            if len(kwargs) == 1:
                listValidate = ["strPath"]
                for kwarg in kwargs :
                    if kwarg not in listValidate :
                        raise TypeError("Keyword argument not understood")

                strPath = kwargs["strPath"]

                self.G = tf.keras.models.load_model(strPath)
                self.D = tf.keras.models.load_model(strPath)

                model : tf.keras.Model = tf.keras.models.load_model(strPath)

                super(ModelTest, self).__init__(strName=model.name, tupleInputShape=tupleInputShape, tupleOutputShape=None)
        
                self.G = NetG(tupleInputShape[1], self.intLatentDim, tupleInputShape[3], self.intDiscriminatorFilter)
                self.D = NetD(tupleInputShape[1], self.intLatentDim, tupleInputShape[3], self.intDiscriminatorFilter)

                self._m_bInit = True

            elif len(kwargs) == 4:
                listValidate = ["strName", "tupleInputShape", "nClasses", "strWeight"]
                for kwarg in kwargs :
                    if kwarg not in listValidate :
                        raise TypeError("Keyword argument not understood")
                    
                strName = kwargs["strName"]
                tupleInputShape = kwargs["tupleInputShape"]
                nClasses = kwargs["nClasses"]
                strWeight = kwargs["strWeight"]

                if tupleInputShape[0] != None : 
                    tupleInputShape = (None, ) + tupleInputShape
                self._m_tupleOutputShape = (None, )+[1]

                self.G = NetG(tupleInputShape[1], self.intLatentDim, tupleInputShape[3], self.intDiscriminatorFilter)
                self.D = NetD(tupleInputShape[1], self.intLatentDim, tupleInputShape[3], self.intDiscriminatorFilter)

                super(ModelTest, self).__init__(strName=strName, tupleInputShape=tupleInputShape, tupleOutputShape=self._m_tupleOutputShape)                
                
                self._m_bInit = True
            break

    def Initialize(self, strName : str, tupleInputShape : tuple, nClasses : int) -> tuple[bool, str]:
        bReturn : bool = False
        strMsg : str = ""
        tupleReturn : tuple = None

        while True :
            if self._m_bInit == True :
                strMsg = "Already initialized"
                break
            if tupleInputShape[0] != None : 
                tupleInputShape = (None, ) + tupleInputShape
            super(ModelTest, self).__init__(strName=strName, tupleInputShape=tupleInputShape,tupleOutputShape=None)

            self.G = NetG(tupleInputShape[1], self.intLatentDim, tupleInputShape[3], self.intDiscriminatorFilter)
            self.D = NetD(tupleInputShape[1], self.intLatentDim, tupleInputShape[3], self.intDiscriminatorFilter)

            self._m_tupleOutputShape = (None, ) + [1]
            self._m_bInit = True
            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg))

        return tupleReturn

    def Terminate(self) -> tuple[bool, str] : 
        bReturn : bool = False
        strMsg : str = ""
        tupleReturn : tuple = None

        while True :
            
            tf.keras.backend.clear_session()
            self._m_bInit = False

            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg))

        return tupleReturn

    def ResetMetrics(self) -> tuple[bool, str] :
        bReturn : bool = False
        strMsg : str = ""
        tupleReturn : tuple = None

        while True :
                        
            self.fGloss = 0
            self.fDloss = 0
            self._m_intImageCount = 0
            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg))

        return tupleReturn
    
    def SaveModel(self, strModelPath : str, bOverWrite : bool) -> tuple[bool, str]:
        bReturn : bool = False
        strMsg : str = ""
        tupleReturn : tuple = None

        while True :
            if self._m_bInit == False :
                strMsg = "not initailze model"
                break
            if strModelPath == "" :
                strMsg = "model path is empty"
                break

            self.G.save(strModelPath, overwrite=bOverWrite)
            self.D.save(strModelPath, overwrite=bOverWrite)

            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg))

        return tupleReturn
    
    def LoadModel(self, strModelPath : str) -> tuple[bool, str]:
        bReturn : bool = False
        strMsg : str = ""
        tupleReturn : tuple = None

        while True :
            if self._m_bInit == False :
                strMsg = "not initailze model"
                break
            if strModelPath == "" :
                strMsg = "model path is empty"
                break
            
            tf.keras.backend.clear_session()

            self.G = tf.keras.models.load_model(strModelPath)
            self.D = tf.keras.models.load_model(strModelPath)
    
            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg))

        return tupleReturn
    
    def TrainBatch(self, listInputBatch : list ,listLabelBatch : list, bResetMetrics : bool = False) -> tuple[bool, str, dict]:
        bReturn : bool = False
        strMsg : str = ""
        dictReturn : dict = {}
        tupleReturn : tuple = None

        while True :

            if self._m_bInit == False :
                strMsg = "not initailze model"
                break
            if len(listInputBatch) == 0 or len(listLabelBatch) == 0  :
                strMsg = "not exist data"
                break

            if bResetMetrics == True :
                self.ResetMetrics()
            
            self._m_intImageCount += 1
            self._m_intBatchsize = len(listInputBatch)
            loss = self.TrainStep(listInputBatch, listLabelBatch)

            self.fGloss += loss[0].numpy()
            self.fDloss += loss[1].numpy()

            dictReturn["loss"] = float((self.fGloss + self.fDloss)/self._m_intImageCount)
            dictReturn["accuracy"] = float()
            dictReturn["precision"] = float()
            dictReturn["recall"] = float()
            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg, dictReturn))

        return tupleReturn
    
    def TestBatch(self, listInputBatch : list ,listLabelBatch : list, bResetMetrics : bool = False) -> tuple[bool, str, dict] :
        bReturn : bool = False
        strMsg : str = ""
        dictReturn : dict = {}
        tupleReturn : tuple = None

        while True :

            if self._m_bInit == False :
                strMsg = "not initailze model"
                break
            if len(listInputBatch) == 0 or len(listLabelBatch) == 0  :
                strMsg = "not exist data"
                break

            if bResetMetrics == True :
                self.ResetMetrics()

            loss = self.ValidStep(listInputBatch, listLabelBatch)

            self.fGloss += loss[0].numpy()
            self.fDloss += loss[1].numpy()
            
            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg, dictReturn))

        return tupleReturn

    def PredictBatch(self, listInputBatch : list) -> tuple[bool, str, list]:
        bReturn : bool = False
        strMsg : str = ""
        dictReturn : dict = {}
        tupleReturn : tuple = None

        while True :

            if self._m_bInit == False :
                strMsg = "not initailze model"
                break
            if len(listInputBatch) == 0 :
                strMsg = "not exist data"
                break

            # npResult = model.predict_on_batch(data)
            
            dictReturn["classProb"] = npResult

            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg, dictReturn))

        return tupleReturn
    
    def CompileModel(self, funcLoss: Callable, funcOptimizer: Callable, listMetrics: list) -> tuple[bool, str]:
        bReturn : bool = False
        strMsg : str = ""
        tupleReturn : tuple = None

        while True :

            if self._m_bInit == False :
                strMsg = "not initailze model"
                break
            if funcLoss == None :
                strMsg = "loss function is none"
                break

            if funcOptimizer == None :
                strMsg = "loss function is none"
                break
            if len(listMetrics) == 0 :
                strMsg = "metric list is empty"
                break

            #self.compile(loss=funcLoss, optimizer=funcOptimizer, metrics=listMetrics)
            
            self.d_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                    beta_1=self.opt.beta1,
                                                    beta_2=0.999)
            self.g_optimizer = tf.keras.optimizers.Adam(self.opt.lr,
                                                        beta_1=self.opt.beta1,
                                                        beta_2=0.999)

            self.adversarialLoss = tf.keras.losses.MeanSquaredError()
            self.contextualLoss = tf.keras.losses.MeanAbsoluteError()
            self.encoderLoss = tf.keras.losses.MeanSquaredError()
            self.discriminatorLoss = tf.keras.losses.BinaryCrossentropy()

            bReturn = True
            break
        
        tupleReturn = tuple((bReturn, strMsg))

        return tupleReturn
    
    def CalculateGLoss(self, image, feat_real, feat_fake, gen_img, latent_i, latent_o):
        err_g_adv = self.adversarialLoss(feat_real, feat_fake)
        err_g_con = self.contextualLoss(image, gen_img)
        err_g_enc = self.encoderLoss(latent_i, latent_o)
        g_loss = err_g_adv * 1 + err_g_con * 50 + err_g_enc * 1  ####### 가중치 변경에 따라 모델 성능 바뀜 이건 Mnist 가중치베스트
    
        return g_loss

    def CalculateDLoss(self, pred_real, pred_fake, real_label, fake_label):
        self.real_label = tf.ones([self._m_intBatchsize], dtype=tf.float32)
        self.fake_label = tf.zeros([self._m_intBatchsize], dtype=tf.float32)

        err_d_real = self.discriminatorLoss(pred_real, real_label)
        err_d_fake = self.discriminatorLoss(pred_fake, fake_label)
        d_loss = (err_d_real + err_d_fake) * 0.5
        return d_loss

    @tf.function
    def TrainStep(self, x):
        """ Autograph enabled by tf.function could speedup more than 6x than eager mode.
        """
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            latent_i, gen_img, latent_o = self.G(x)
            pred_real, feat_real = self.D(x)
            pred_fake, feat_fake = self.D(gen_img)
            g_loss = self.CalculateGLoss(x, feat_real, feat_fake, gen_img, latent_i, latent_o)
            d_loss = self.CalculateDLoss(pred_real, pred_fake)

        g_grads = g_tape.gradient(g_loss, self.G.trainable_weights)
        d_grads = d_tape.gradient(d_loss, self.D.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.G.trainable_weights))
        self.d_optimizer.apply_gradients(zip(d_grads, self.D.trainable_weights))
        return g_loss, d_loss
    
    @tf.function
    def ValidStep(self, x, y):
        latent_i, gen_img, latent_o = self.G(x)
        latent_i, gen_img, latent_o = latent_i.numpy(), gen_img.numpy(), latent_o.numpy()

        error = np.mean((latent_i - latent_o)**2, axis=-1)
        an_scores.append(error)
        gt_labels.append(y)
        an_scores = np.concatenate(an_scores, axis=0).reshape([-1])
        gt_labels = np.concatenate(gt_labels, axis=0).reshape([-1])

        g_loss = self.CalculateGLoss()
        d_loss = self.CalculateDLoss()

        return an_scores, gt_labels, g_loss, d_loss
