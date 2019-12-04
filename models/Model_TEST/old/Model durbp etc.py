from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *

from facelib import PoseEstimator

class TrueFaceModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_face_type = 'f'
        if is_first_run:
            self.options['resolution'] = io.input_int("分辨率 ( 128，256 帮助:? 跳过:128) : ", 128, [128,256], help_message="更高的分辨率需要更多的VRAM和训练时间。 数值调整成16的倍数。")
        else:
            self.options['resolution'] = self.options.get('resolution', 128)


    #override
    def onInitialize(self, batch_size=-1, **in_options):


        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({4:8})

        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        bgrm_shape = (resolution, resolution, 4)

        ngf = 64
        ndf = 64
        lambda_A = 50
        lambda_B = 50

        use_batch_norm = True #created_batch_size > 1



        # self.enc = modelify(TrueFaceModel.DFEncFlow ()) ( Input(bgr_shape) )
        # dec_Inputs = [ Input(K.int_shape(x)[1:]) for x in self.enc.outputs ]
        # self.decA = modelify(TrueFaceModel.DFDecFlow (bgr_shape[2])) (dec_Inputs)
        # self.decB = modelify(TrueFaceModel.DFDecFlow (bgr_shape[2])) (dec_Inputs)

        # def GA(x):
        #     return self.decA(self.enc(x))
        # self.GA = GA
        # def GB(x):
        #     return self.decB(self.enc(x))
        # self.GB = GB


        self.GA = modelify(TrueFaceModel.ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf))( Input(bgr_shape) )
        self.GB = modelify(TrueFaceModel.ResNet (bgr_shape[2], use_batch_norm, n_blocks=6, ngf=ngf))( Input(bgr_shape) )

        #self.GA = modelify(TrueFaceModel.UNet (bgr_shape[2], use_batch_norm, ngf=ngf))(Input(bgr_shape))
        #self.GB = modelify(TrueFaceModel.UNet (bgr_shape[2], use_batch_norm, ngf=ngf))(Input(bgr_shape))

        #self.GA = modelify(TrueFaceModel.DURNUFlow (bgr_shape[2]))(Input(bgr_shape))
        #self.GB = modelify(TrueFaceModel.DURNUFlow (bgr_shape[2]))(Input(bgr_shape))
        

        self.DA = modelify(TrueFaceModel.PatchDiscriminator(ndf=ndf) ) ( Input(bgr_shape) )
        self.DB = modelify(TrueFaceModel.PatchDiscriminator(ndf=ndf) ) ( Input(bgr_shape) )

        if not self.is_first_run():
            weights_to_load = [
                (self.GA, 'GA.h5'),
                (self.GB, 'GB.h5'),
                # (self.enc, 'enc.h5'),
                # (self.decA, 'decA.h5'),
                # (self.decB, 'decB.h5'),
                (self.DA, 'DA.h5'),
                (self.DB, 'DB.h5'),
            ]
            self.load_weights_safe(weights_to_load)

        real_A0 = Input(bgr_shape)
        real_A0m = Input(mask_shape)
        real_B0 = Input(bgr_shape)
        real_B0m = Input(mask_shape)

        real_A0_pyr = Input ( (3,) )
        real_B0_pyr = Input ( (3,) )

        def DLoss(labels,logits):
            return K.mean(K.binary_crossentropy(labels,logits))


        fake_B0 = self.GA(real_A0)
        fake_A0 = self.GB(real_B0)


        real_A0_d = self.DA(real_A0)
        real_A0_d_ones = K.ones_like(real_A0_d)


        fake_A0_d  = self.DA(fake_A0)
        fake_A0_d_ones = K.ones_like(fake_A0_d)
        fake_A0_d_zeros = K.zeros_like(fake_A0_d)

        real_B0_d = self.DB(real_B0)
        real_B0_d_ones = K.ones_like(real_B0_d)

        fake_B0_d = self.DB(fake_B0)
        fake_B0_d_ones = K.ones_like(fake_B0_d)
        fake_B0_d_zeros = K.zeros_like(fake_B0_d)

        rec_A0 = self.GB (fake_B0)
        rec_B0 = self.GA (fake_A0)

        loss_GA = DLoss(fake_B0_d_ones, fake_B0_d ) + \
                  K.mean ( 50 * dssim(kernel_size=int(resolution/11.6),max_value=2.0)(rec_B0+1,real_B0+1) ) + \
                  K.mean ( 34 * K.square (rec_B0-real_B0))
                  

        weights_GA = self.GA.trainable_weights #self.enc.trainable_weights + self.decA.trainable_weights# + 

        loss_GB = DLoss(fake_A0_d_ones, fake_A0_d ) + \
                  K.mean ( 50 * dssim(kernel_size=int(resolution/11.6),max_value=2.0)(rec_A0+1,real_A0+1) ) + \
                  K.mean ( 34 * K.square (rec_A0-real_A0))
                  
        weights_GB = self.GB.trainable_weights #self.enc.trainable_weights + self.decB.trainable_weights

        def opt():
            return Adam(lr=5e-5, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)

        self.GA_train = K.function ([real_A0, real_A0m, real_B0, real_B0m],[loss_GA],
                                    opt().get_updates(loss_GA, weights_GA) )

        self.GB_train = K.function ([real_A0, real_A0m, real_B0, real_B0m],[loss_GB],
                                    opt().get_updates(loss_GB, weights_GB) )


        ########### DLoss(real_A0_ident_pred, K.zeros_like(real_A0_ident_pred) ) +\

        loss_D_A = ( DLoss(real_A0_d_ones,  real_A0_d ) + \
                     DLoss(fake_A0_d_zeros, fake_A0_d ) ) * 0.5 
                     #10 * K.mean(K.square(real_A0_pyr_pred - real_A0_pyr))


        self.DA_train = K.function ([real_A0, real_A0m, real_B0, real_B0m],[loss_D_A],
                                    opt().get_updates(loss_D_A, self.DA.trainable_weights) )

        ############ DLoss(real_B0_ident_pred, K.ones_like(real_B0_ident_pred) ) +\

        loss_D_B = ( DLoss(real_B0_d_ones,  real_B0_d ) + \
                     DLoss(fake_B0_d_zeros, fake_B0_d ) ) * 0.5 
                     #10 * K.mean(K.square(real_B0_pyr_pred - real_B0_pyr))

        self.DB_train = K.function ([real_A0, real_A0m, real_B0, real_B0m],[loss_D_B],
                                    opt().get_updates(loss_D_B, self.DB.trainable_weights) )

        ############


        self.G_view = K.function([real_A0, real_A0m, real_B0, real_B0m],[fake_A0, rec_A0, fake_B0, rec_B0 ])



        if self.is_training_mode:
            
            if self.is_first_run():
                conv_weights_list = []
                for model in [self.GA, self.GB, self.DA, self.DB]:
                    for layer in model.layers:
                        if type(layer) == keras.layers.Conv2D:
                            conv_weights_list += [layer.weights[0]] #Conv2D kernel_weights
                CAInitializerMP ( conv_weights_list )
            
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            output_sample_types=[ {'types': (t.IMG_SOURCE, face_type, t.MODE_BGR), 'resolution':resolution, 'normalize_tanh':True},
                                  {'types': (t.IMG_SOURCE, face_type, t.MODE_M), 'resolution':resolution, 'normalize_tanh':True},
                                ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types=output_sample_types )
                   ])
        else:
            self.G_convert = K.function([real_B0],[fake_A0])

    #override
    def onSave(self):
        self.save_weights_safe( [
                                 [self.GA,   'GA.h5'],
                                 [self.GB,   'GB.h5'],
                                #  [self.enc,   'enc.h5'],
                                #  [self.decA,   'decA.h5'],
                                #  [self.decB,   'decB.h5'],
                                 [self.DA,   'DA.h5'],
                                 [self.DB,   'DB.h5']
                                 ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        src, srcm, = generators_samples[0]
        dst, dstm, = generators_samples[1]

        feed = [src, srcm, dst, dstm, ]

        loss_GA, = self.GA_train ( feed )
        loss_GB, = self.GB_train ( feed )
        loss_DA, = self.DA_train( feed )
        loss_DB, = self.DB_train( feed )

        return ( ('GA', loss_GA), ('GB', loss_GB), ('DA', loss_DA),  ('DB', loss_DB)  )

    #override
    def onGetPreview(self, sample):
        test_A0   = sample[0][0][0:4]
        test_A0m  = sample[0][1][0:4]

        test_B0   = sample[1][0][0:4]
        test_B0m  = sample[1][1][0:4]

        G_view_result = self.G_view([test_A0, test_A0m, test_B0, test_B0m])

        fake_A0, rec_A0, fake_B0, rec_B0 = [ x[0] / 2 + 0.5 for x in G_view_result]
        test_A0, test_A0m, test_B0, test_B0m = [ x[0] / 2 + 0.5 for x in [test_A0, test_A0m, test_B0, test_B0m] ]

        r = np.concatenate ((np.concatenate ( (test_A0, fake_B0, rec_A0), axis=1),
                             np.concatenate ( (test_B0, fake_A0, rec_B0), axis=1)
                             ), axis=0)

        return [ ('TrueFace', r ) ]

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.G_convert ([ np.zeros ( (1, self.options['resolution'], self.options['resolution'], 3), dtype=np.float32 ) ])
        else:
            bgr, = self.G_convert ([  face[np.newaxis,...]*2-1  ]) 
            return bgr[0] / 2 + 0.5

    #override
    def get_ConverterConfig(self):
        face_type = FaceType.FULL

        import converters
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), converters.ConverterConfigMasked(face_type=face_type,
                                     default_mode = 1,
                                     clip_hborder_mask_per=0.0625 if (face_type == FaceType.FULL) else 0,
                                    )
                               
    @staticmethod
    def DURNUFlow(output_nc=3):

        def DURBUFlow(dim, res_dim, f_size=3, dilation=1):

            def func(x, res):
                x_r = x

                x = ReLU()(InstanceNormalization(axis=-1)( Conv2D(dim, kernel_size=3, strides=1, padding='same')(x) ))
                x = Conv2D(dim, kernel_size=3, strides=1, padding='same')(x)

                x = Add()([x,x_r])
                x = ReLU()(InstanceNormalization(axis=-1)(x))

                x = SubpixelUpscaler()(InstanceNormalization(axis=-1)(Conv2D(res_dim*4, kernel_size=1, strides=1, padding='same')(x) ))

                x = Conv2D(res_dim, kernel_size=f_size, strides=1, padding='same', dilation_rate=dilation)(x)
                x = Add()([x,res])
                x = ReLU()(InstanceNormalization(axis=-1)(x))
                res = x

                x = Conv2D(dim, kernel_size=3, strides=2, padding='same')(x)
                x = Add()([x,x_r])
                x = ReLU()(InstanceNormalization(axis=-1)(x))
                return x, res

            return func

        def func(inp):
            x = inp
            x =       ReLU()(InstanceNormalization(axis=-1)( Conv2D(64,  kernel_size=7, strides=1, padding='same')(x) ))
            res = x = ReLU()(InstanceNormalization(axis=-1)( Conv2D(128, kernel_size=3, strides=2, padding='same')(x) ))
            x =       ReLU()(InstanceNormalization(axis=-1)( Conv2D(256, kernel_size=3, strides=2, padding='same')(x) ))

            x, res = DURBUFlow(256, 128, f_size=3, dilation=3)(x, res)
            x, res = DURBUFlow(256, 128, f_size=7, dilation=1)(x, res)
            x, res = DURBUFlow(256, 128, f_size=3, dilation=3)(x, res)
            x, res = DURBUFlow(256, 128, f_size=7, dilation=1)(x, res)
            x, res = DURBUFlow(256, 128, f_size=3, dilation=2)(x, res)
            x, res = DURBUFlow(256, 128, f_size=5, dilation=1)(x, res)

            x = SubpixelUpscaler()(InstanceNormalization(axis=-1)( Conv2D(128*4, kernel_size=1, strides=1, padding='same')(x) ))
            x = ReLU()(InstanceNormalization(axis=-1)( Conv2D(128,  kernel_size=3, strides=1, padding='same')(x) ))

            x = SubpixelUpscaler()(InstanceNormalization(axis=-1)( Conv2D(64*4, kernel_size=1, strides=1, padding='same')(x) ))
            x = ReLU()(InstanceNormalization(axis=-1)( Conv2D(64, kernel_size=3, strides=1, padding='same')(x) ))

            x = Conv2D(output_nc, kernel_size=7, strides=1, padding='same', activation='tanh')(x)
            return x

        return func
        
    @staticmethod
    def ResNet(output_nc, use_batch_norm, ngf=64, n_blocks=6, use_dropout=False):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, padding='same', use_bias=use_bias)
        XConv2DTranspose = partial(Conv2DTranspose, padding='same', use_bias=use_bias)

        def func(input):


            def ResnetBlock(dim, use_dropout=False):
                def func(input):
                    x = input

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)
                    x = ReLU()(x)

                    if use_dropout:
                        x = Dropout(0.5)(x)

                    x = XConv2D(dim, 3, strides=1)(x)
                    x = XNormalization(x)

                    return Add()([x,input])
                return func

            x = input

            x = XConv2D(ngf, 7, strides=1, use_bias=True)(x)

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))
            
            x = ReLU()(XNormalization(XConv2D(ngf*8, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*8, use_dropout=use_dropout)(x)

            x = ReLU()(XNormalization(XConv2DTranspose(ngf*4, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='tanh', use_bias=True)(x)

            return x

        return func

    @staticmethod
    def PatchDiscriminator(ndf=64):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        def func(input):

            x = input


            x = XConv2D( ndf, 4, strides=2, padding='same', use_bias=True)(x) #4
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*2, 3, strides=2, padding='same')(x) #8
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*4, 3, strides=2, padding='same')(x) #16
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = XConv2D( ndf*8, 3, strides=2, padding='same')(x) #32
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = XConv2D( ndf*8, 3, strides=2, padding='same')(x) #64
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            x = XConv2D( ndf*8, 3, strides=2, padding='same')(x) #128
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            return XConv2D( 1, 3, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)
        return func

    @staticmethod
    def NLayerDiscriminator(use_batch_norm, ndf=64, n_layers=3):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=True)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            f = ndf

            x = XConv2D( f, 3, strides=1, padding='same')(x)
            x = XConv2D( f, 3, strides=2, padding='same', activation='relu')(x)
            f = min( ndf*8, f*2 )

            for i in range(n_layers):
                x = XConv2D( f, 3, strides=1, padding='same')(x)
                x = XConv2D( f, 3, strides=2, padding='same', activation='relu')(x)
                f = min( ndf*8, f*2 )

            x = XConv2D( f, 3, strides=1, padding='same')(x)
            x = XConv2D( f, 3, strides=1, padding='same')(x)
            x = XConv2D( 128, 3, strides=2, padding='same' )(x)

            out1 = XConv2D( 1, 4, strides=1, padding='same', activation='sigmoid')(x)

            return [out1]
        return func


Model = TrueFaceModel
