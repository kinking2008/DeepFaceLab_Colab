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

class AVATARModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        def_resolution = 128
        if is_first_run:
            self.options['resolution'] = io.input_int("分辨率 ( 128,256 帮助:? 跳过:%d) : " % def_resolution, def_resolution, [128,256], help_message="更高的分辨率需要更多的VRAM和训练时间。 数值调整成16的倍数。")
        else:
            self.options['resolution'] = self.options.get('resolution', def_resolution)


    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({2:1})
        AVATARModel.initialize_nn_functions()
        
        resolution = self.options['resolution']
        in_bgr_shape = (64, 64, 3)
        bgr_64_mask_shape = (64,64,1)
        out_bgr_shape = (resolution, resolution, 3)
        bgr_t_shape = (resolution, resolution, 9)
        mask_shape = (resolution, resolution, 1)
        bgrm_shape = (resolution, resolution, 4)

        ngf = 64
        ndf = 64

        use_batch_norm = True
        
        self.enc = modelify(AVATARModel.EncFlow())( [Input(in_bgr_shape),] )
        self.decA64 = modelify(AVATARModel.Dec64Flow()) ( [ Input(K.int_shape(self.enc.outputs[0])[1:]) ] )        
        self.decB64 = modelify(AVATARModel.Dec64Flow()) ( [ Input(K.int_shape(self.enc.outputs[0])[1:]) ] )
        
        #self.GA = modelify(AVATARModel.ResNet (in_bgr_shape[2], use_batch_norm=False, ngf=64, use_dropout=True))(Input(in_bgr_shape))
        #self.GB = modelify(AVATARModel.ResNet (in_bgr_shape[2], use_batch_norm=False, ngf=64, use_dropout=True))(Input(in_bgr_shape))
        
        self.DA = modelify(AVATARModel.D64Discriminator() ) (Input(out_bgr_shape))
        self.DB = modelify(AVATARModel.D64Discriminator() ) (Input(out_bgr_shape))

        
        self.C = modelify(AVATARModel.ResNet (9, use_batch_norm=False, n_blocks=6, ngf=128, use_dropout=True))(Input(bgr_t_shape))
        
        if not self.is_first_run():
            weights_to_load = [
                [self.enc, 'enc.h5'],                
                [self.decA64, 'decA64.h5'],
                [self.decB64, 'decB64.h5'],
                [self.DA, 'DA.h5'],
                [self.DB, 'DB.h5'],
                [self.C, 'C.h5']
            ]
            self.load_weights_safe(weights_to_load)

        DA_ones =  K.ones_like ( K.shape(self.DA.outputs[0]) )
        DA_zeros = K.zeros_like ( K.shape(self.DA.outputs[0] ))
        DB_ones = K.ones_like ( K.shape(self.DB.outputs[0] ))
        DB_zeros = K.zeros_like ( K.shape(self.DB.outputs[0] ))
        def DLoss(labels,logits):
            return K.mean(K.binary_crossentropy(labels,logits))


        real_A_in = Input(in_bgr_shape)
        real_Am = Input(bgr_64_mask_shape)
        real_B_in = Input(in_bgr_shape)
        real_Bm = Input(bgr_64_mask_shape)
        
        real_A = ( (real_A_in+1)*real_Am + (1-real_Am) ) -1
        real_B = ( (real_B_in+1)*real_Bm + (1-real_Bm) ) -1
        
        fake_B = self.decA64(self.enc(real_A))       
        fake_A = self.decB64(self.enc(real_B))
        
        rec_A = self.decB64(self.enc(fake_B))        
        rec_B = self.decA64(self.enc(fake_A)) 
         
        real_A_d = self.DA(real_A)
        real_A_d_ones = K.ones_like(real_A_d)        
        fake_A_d = self.DA(fake_A)
        fake_A_d_ones = K.ones_like(fake_A_d)
        fake_A_d_zeros = K.zeros_like(fake_A_d)
        real_B_d = self.DB(real_B)
        real_B_d_ones = K.ones_like(real_B_d)                
        fake_B_d = self.DB(fake_B)
        fake_B_d_ones = K.ones_like(fake_B_d)
        fake_B_d_zeros = K.zeros_like(fake_B_d)

        
        
        real_A_t0 = Input(out_bgr_shape)
        real_Am_t0 = Input(mask_shape)
        real_A_t1 = Input(out_bgr_shape)
        real_Am_t1 = Input(mask_shape)
        real_A_t2 = Input(out_bgr_shape)
        real_Am_t2 = Input(mask_shape)

        real_B_t0_in = Input(out_bgr_shape)
        real_Bm_t0   = Input(mask_shape)
        real_B_t1_in = Input(out_bgr_shape)
        real_Bm_t1   = Input(mask_shape)
        real_B_t2_in = Input(out_bgr_shape)
        real_Bm_t2   = Input(mask_shape)
        real_B_t0 = real_B_t0_in*real_Bm_t0 + (1-real_Bm_t0)*0.5
        real_B_t1 = real_B_t1_in*real_Bm_t1 + (1-real_Bm_t1)*0.5
        real_B_t2 = real_B_t2_in*real_Bm_t2 + (1-real_Bm_t2)*0.5
        
        x = self.C ( K.concatenate ( [real_A_t0*real_Am_t0 + (1-real_Am_t0)*0.5,
                                      real_A_t1*real_Am_t1 + (1-real_Am_t1)*0.5,
                                      real_A_t2*real_Am_t2 + (1-real_Am_t2)*0.5
                                     ] , axis=-1) )
        rec_C_A_t0 = Lambda ( lambda x: x[...,0:3], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_A_t1 = Lambda ( lambda x: x[...,3:6], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_A_t2 = Lambda ( lambda x: x[...,6:9], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)

        rec_AB_t0 = self.decB64(self.enc(real_B_t0))
        rec_AB_t1 = self.decB64(self.enc(real_B_t1))
        rec_AB_t2 = self.decB64(self.enc(real_B_t2))
        
        x = self.C ( K.concatenate ( [rec_AB_t0, rec_AB_t1, rec_AB_t2] , axis=-1) )
        rec_C_AB_t0 = Lambda ( lambda x: x[...,0:3], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_AB_t1 = Lambda ( lambda x: x[...,3:6], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)
        rec_C_AB_t2 = Lambda ( lambda x: x[...,6:9], output_shape= ( K.int_shape(x)[1:3], 3 ) ) (x)

 
        
        if self.is_training_mode:
            def opt(lr=2e-5):
                return Adam(lr=lr, beta_1=0.5, beta_2=0.999, tf_cpu_mode=2)#, clipnorm=1)
                
            def CycleLoss(t1,t2):
                return 50 * K.mean(K.abs(t1 - t2))
                #return K.mean( 50 * dssim(kernel_size=int(resolution/11.6),max_value=1.0)(t1, t2) )
                
            loss_GA = DLoss(fake_B_d_ones, fake_B_d ) + CycleLoss(rec_B, real_B)
            weights_GA = self.enc.trainable_weights + self.decA64.trainable_weights
            loss_GB = DLoss(fake_A_d_ones, fake_A_d ) + CycleLoss(rec_A, real_A)
            weights_GB = self.enc.trainable_weights + self.decB64.trainable_weights
        
            self.GA_train = K.function ([real_A_in, real_Am, real_B_in, real_Bm],[loss_GA],
                                        opt().get_updates(loss_GA, weights_GA) )
                                    
            self.GB_train = K.function ([real_A_in, real_Am, real_B_in, real_Bm],[loss_GB],
                                        opt().get_updates(loss_GB, weights_GB) )
        
            ###########        
        
            loss_D_A = (DLoss(real_A_d_ones, real_A_d ) + \
                        DLoss(fake_A_d_zeros, fake_A_d ) ) * 0.5

            self.DA_train = K.function ([real_A_in, real_Am, real_B_in, real_Bm],[loss_D_A],
                                        opt().get_updates(loss_D_A, self.DA.trainable_weights) )
            
            ############
            
            loss_D_B = (DLoss(real_B_d_ones, real_B_d ) + \
                        DLoss(fake_B_d_zeros, fake_B_d ) ) * 0.5
            
            self.DB_train = K.function ([real_A_in, real_Am, real_B_in, real_Bm],[loss_D_B],
                                        opt().get_updates(loss_D_B, self.DB.trainable_weights) )
            
            ############
        
            self.G_view = K.function([real_A_in, real_Am, real_B_in, real_Bm], [fake_A, fake_B, rec_A, rec_B, ])#, real_B_t0_in, real_Bm_t0, real_B_t1_in, real_Bm_t1, real_B_t2_in, real_Bm_t2      rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2
    
            loss_C = K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t0, rec_C_A_t0 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t1, rec_C_A_t1 ) ) + \
                     K.mean( 10 * dssim(kernel_size=int(resolution/11.6),max_value=1.0) ( real_A_t2, rec_C_A_t2 ) ) 
                 
            weights_C = self.C.trainable_weights
            
            self.C_train = K.function ([ real_A_t0, real_Am_t0, real_A_t1, real_Am_t1, real_A_t2, real_Am_t2 ],[ loss_C ], opt().get_updates(loss_C, weights_C) )
            ###########

            t = SampleProcessor.Types

            output_sample_types=[ {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64, 'normalize_tanh':True},
                                  {'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_M), 'resolution':64}
                                ]

            generators = [
                                           
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0]),
                        output_sample_types=output_sample_types ),
                        
                    SampleGeneratorFaceTemporal(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False), 
                        output_sample_types=[{'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                             {'types': (t.IMG_WARPED_TRANSFORMED, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_M), 'resolution':64},
                                             {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                            ] ),
                       
                    SampleGeneratorFaceTemporal(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                        temporal_image_count=3,
                        sample_process_options=SampleProcessor.Options(random_flip=False), 
                        output_sample_types=[{'types': (t.IMG_SOURCE, t.FACE_TYPE_FULL_NO_ROTATION, t.MODE_BGR), 'resolution':64},
                                             {'types': (t.IMG_SOURCE, t.NONE, t.MODE_BGR), 'resolution':resolution},
                                            ] ),                                             
                   ]
            generators[2].set_active(False)
            generators[3].set_active(False)
            self.set_training_data_generators (generators)
        else:
            self.G_convert = K.function([warped_B064],[rec_C_A0_B0])

    def get_model_filename_list(self):
        return [[self.enc, 'enc.h5'],                
                [self.decA64, 'decA64.h5'],
                [self.decB64, 'decB64.h5'],
                [self.DA, 'DA.h5'],
                [self.DB, 'DB.h5'],
                [self.C, 'C.h5']
              ]
               
    #override
    def onSave(self):
        self.save_weights_safe( [  [self.enc, 'enc.h5'],                
                                    [self.decA64, 'decA64.h5'],
                                    [self.decB64, 'decB64.h5'],
                                    [self.DA, 'DA.h5'],
                                    [self.DB, 'DB.h5'],
                                    [self.C, 'C.h5']
                                ])

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        src, srcm, = generators_samples[0]
        dst, dstm, = generators_samples[1]                
        #t_src_0, t_srcm_0, t_src_1, t_srcm_1, t_src_2, t_srcm_2, = generators_samples[2]

        feed = [src, srcm, dst, dstm]
        loss_GA, = self.GA_train (feed)
        loss_GB, = self.GB_train (feed)
        loss_DA, = self.DA_train (feed)
        loss_DB, = self.DB_train (feed)
        
        loss_C, = 0,#self.C_train ( [ t_src_0, t_srcm_0, t_src_1, t_srcm_1, t_src_2, t_srcm_2 ] )

        return ( ('GA', loss_GA), ('GB', loss_GB), ('DA', loss_DA), ('DB', loss_DB), ('C', loss_C) )

    #override
    def onGetPreview(self, sample):
        real_A  = sample[0][0][0:4]
        real_Am  = sample[0][1][0:4]

        real_B  = sample[1][0][0:4]
        real_Bm  = sample[1][1][0:4]

        #t_dst_0   = sample[3][0][0:4]
        #t_dstm_0  = sample[3][1][0:4]
        #t_dst_1   = sample[3][2][0:4]
        #t_dstm_1  = sample[3][3][0:4]
        #t_dst_2   = sample[3][4][0:4]
        #t_dstm_2  = sample[3][5][0:4]
        
  
        G_view_result = self.G_view([real_A, real_Am, real_B, real_Bm])#, t_dst_0, t_dstm_0, t_dst_1, t_dstm_1, t_dst_2, t_dstm_2, ])

        real_A, real_B, fake_A, fake_B, rec_A, rec_B,  = [ (x[0]+1)/2 for x in ([real_A, real_B, ] + G_view_result)  ]#rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2

        #r = sample64x4
        l1 = np.concatenate ( (real_A, fake_B, rec_A), axis=1 )
        l2 = np.concatenate ( (real_B, fake_A, rec_B), axis=1 )
        #l3 = np.concatenate ( (t_dst_0, t_dst_1, t_dst_2), axis=1 )
        #l4 = np.concatenate ( (rec_C_AB_t0, rec_C_AB_t1, rec_C_AB_t2 ), axis=1 )
        #r = np.concatenate ( (l1,l2,l3,l4), axis=0)
        r = np.concatenate ( (l1,l2), axis=0)
        
        return [ ('AVATAR', r ) ]

    def predictor_func (self, inp_f0, inp_f1, inp_f2):        
        feed = [ inp_f0[np.newaxis,...], inp_f1[np.newaxis,...], inp_f2[np.newaxis,...] ]
        x = self.G_convert (feed)[0]
        return np.clip ( x[0], 0, 1)

    # #override
    # def get_converter(self, **in_options):
    #     from models import ConverterImage
    #     return ConverterImage(self.predictor_func,
    #                           predictor_input_size=self.options['resolution'],
    #                           **in_options)
    #override
    def get_converter(self):
        base_erode_mask_modifier = 30
        base_blur_mask_modifier = 0

        default_erode_mask_modifier = 0
        default_blur_mask_modifier = 0

        face_type = FaceType.FULL

        from converters import ConverterAvatar
        return ConverterAvatar(self.predictor_func,
                               predictor_input_size=64)


    @staticmethod
    def PatchDiscriminator(ndf=64, use_batch_norm=True):
        exec (nnlib.import_all(), locals(), globals())

        if not use_batch_norm:
            use_bias = True
            def XNormalization(x):
                return InstanceNormalization (axis=-1)(x)
        else:
            use_bias = False
            def XNormalization(x):
                return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        """
        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf, 4, strides=2, padding='valid', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*2, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*4, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*8, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            x = XConv2D( ndf*8, 4, strides=2, padding='valid')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = ZeroPadding2D((1,1))(x)
            return XConv2D( 1, 4, strides=1, padding='valid', use_bias=True, activation='sigmoid')(x)#
        """
        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = XConv2D( ndf, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*2, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*4, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*8, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)
            
            x = XConv2D( ndf*16, 4, strides=2, padding='same')(x)
            x = LeakyReLU(0.2)(x)
            x = XNormalization(x)

            return XConv2D( 1, 5, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
        return func
   
    @staticmethod
    def EncFlow(padding='zero', **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        use_bias = False
        def XNorm(x):
            return BatchNormalization (axis=-1)(x)
        XConv2D = partial(Conv2D, padding=padding, use_bias=use_bias)

        def downscale (dim):
            def func(x):
                return LeakyReLU(0.1)( Conv2D(dim, 5, strides=2, padding='same')(x))
            return func

        def upscale (dim):
            def func(x):
                return SubpixelUpscaler()(LeakyReLU(0.1)(Conv2D(dim * 4, 3, strides=1, padding='same')(x)))
            return func

               
        def func(input):
            x, = input
            b,h,w,c = K.int_shape(x)
            x = downscale(64)(x)
            x = downscale(128)(x)
            x = downscale(256)(x)
            x = downscale(512)(x)

            x = Dense(512)(Flatten()(x))
            x = Dense(4 * 4 * 512)(x)
            x = Reshape((4, 4, 512))(x) 
            x = upscale(512)(x)   
            return x
            
        return func

    @staticmethod
    def Dec64Flow(output_nc=3, **kwargs):
        exec (nnlib.import_all(), locals(), globals())

        ResidualBlock = AVATARModel.ResidualBlock
        upscale = AVATARModel.upscale
        to_bgr = AVATARModel.to_bgr

        def func(input):
            x = input[0]
            
            x = upscale(512)(x)
            x = upscale(256)(x)
            x = upscale(128)(x)
            return to_bgr(output_nc, activation="tanh") (x)

        return func
        
    @staticmethod
    def D64Discriminator(ndf=256):
        exec (nnlib.import_all(), locals(), globals())

        #use_bias = True
        #def XNormalization(x):
        #    return InstanceNormalization (axis=-1)(x)
        use_bias = False
        def XNormalization(x):
            return BatchNormalization (axis=-1)(x)

        XConv2D = partial(Conv2D, use_bias=use_bias)

        def func(input):
            b,h,w,c = K.int_shape(input)

            x = input

            x = XConv2D( ndf, 4, strides=2, padding='same', use_bias=True)(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*2, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)

            x = XConv2D( ndf*4, 4, strides=2, padding='same')(x)
            x = XNormalization(x)
            x = LeakyReLU(0.2)(x)
            
            return XConv2D( 1, 4, strides=1, padding='same', use_bias=True, activation='sigmoid')(x)#
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
                    x = ReLU()(x)
                    return Add()([x,input])
                return func

            x = input

            x = ReLU()(XNormalization(XConv2D(ngf, 7, strides=1)(x)))

            x = ReLU()(XNormalization(XConv2D(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2D(ngf*4, 3, strides=2)(x)))

            for i in range(n_blocks):
                x = ResnetBlock(ngf*4, use_dropout=use_dropout)(x)

            x = ReLU()(XNormalization(XConv2DTranspose(ngf*2, 3, strides=2)(x)))
            x = ReLU()(XNormalization(XConv2DTranspose(ngf  , 3, strides=2)(x)))

            x = XConv2D(output_nc, 7, strides=1, activation='tanh', use_bias=True)(x)

            return x

        return func
        
    @staticmethod
    def initialize_nn_functions():
        exec (nnlib.import_all(), locals(), globals())

        class ResidualBlock(object):
            def __init__(self, filters, kernel_size=3, padding='zero', **kwargs):
                self.filters = filters
                self.kernel_size = kernel_size
                self.padding = padding

            def __call__(self, inp):
                x = inp
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = LeakyReLU(0.2)(x)
                x = Conv2D(self.filters, kernel_size=self.kernel_size, padding=self.padding)(x)
                x = Add()([x, inp])
                x = LeakyReLU(0.2)(x)
                return x
        AVATARModel.ResidualBlock = ResidualBlock

        def downscale (dim, padding='zero', act='', **kwargs):
            def func(x):
                return LeakyReLU(0.2) (Conv2D(dim, kernel_size=5, strides=2, padding=padding)(x))
            return func
        AVATARModel.downscale = downscale

        def upscale (dim, padding='zero', norm='', act='', **kwargs):
            def func(x):
                return SubpixelUpscaler()( LeakyReLU(0.2)(Conv2D(dim * 4, kernel_size=3, strides=1, padding=padding)(x)))
            return func
        AVATARModel.upscale = upscale

        def to_bgr (output_nc, padding='zero', activation='sigmoid', **kwargs):
            def func(x):
                return Conv2D(output_nc, kernel_size=5, padding=padding, activation=activation)(x)
            return func
        AVATARModel.to_bgr = to_bgr
              
        
Model = AVATARModel

""" 
def BCELoss(logits, ones):
    if ones:
        return K.mean(K.binary_crossentropy(K.ones_like(logits),logits))
    else:
        return K.mean(K.binary_crossentropy(K.zeros_like(logits),logits))

def MSELoss(labels,logits):
    return K.mean(K.square(labels-logits))

def DLoss(labels,logits):
    return K.mean(K.binary_crossentropy(labels,logits))

def MAELoss(t1,t2):
    return dssim(kernel_size=int(resolution/11.6),max_value=2.0)(t1+1,t2+1 )
    return K.mean(K.abs(t1 - t2) )
"""