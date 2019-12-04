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

class FUNITModel(ModelBase):

    AdaptiveInstanceNormalization2D = None
    
            
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
        self.set_vram_batch_requirements({4:16})

        resolution = self.options['resolution']
        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)
        bgrm_shape = (resolution, resolution, 4)
        label_shape = (1,)

        ngf = 64
        ndf = 64
        latent_dim = 64

        self.enc_class_model = modelify ( FUNITModel.ClassModelEncoderFlow(downs=4, ngf=ngf, latent_dim=latent_dim) ) ( Input(bgr_shape) )
        self.enc_content     = modelify ( FUNITModel.ContentEncoderFlow(downs=2, ngf=ngf, n_res_blks=2) )             ( Input(bgr_shape) )

        self.decoder     = modelify ( FUNITModel.DecoderFlow(ups=2, n_res_blks=2, nf_mlp=256, n_mlp_blks=2  ) ) \
                             ( [ Input(K.int_shape(self.enc_content.outputs[0])[1:], name="decoder_input_1"),
                                 Input(K.int_shape(self.enc_class_model.outputs[0])[1:], name="decoder_input_2")
                               ] )

        self.dis = modelify ( FUNITModel.DiscriminatorFlow(self.batch_size, nf=ndf, n_res_blks=10, num_classes=2) ) (Input(bgr_shape))

        self.G_opt = RMSprop(lr=0.0001, decay=0.0001, tf_cpu_mode=2 if 'tensorflow' in self.device_config.backend else 0)
        self.D_opt = RMSprop(lr=0.0001, decay=0.0001, tf_cpu_mode=2 if 'tensorflow' in self.device_config.backend else 0)

        xa = Input(bgr_shape, name="xa")
        la = Input(label_shape, dtype=np.int32, name="la")

        xb = Input(bgr_shape, name="xb")
        lb = Input(label_shape, dtype=np.int32, name="lb")

        s_xa_one = Input( (self.enc_class_model.outputs[0].shape[-1].value,), name="s_xa_input")

        c_xa = self.enc_content(xa)

        s_xa = self.enc_class_model(xa)
        s_xb = self.enc_class_model(xb)

        s_xa_mean = K.mean(s_xa, axis=0)

        #decoder_updates = []
        #for layer in self.decoder.layers:
        #    if type(layer) == FUNITAdain:        
        #        #import code
        #        #code.interact(local=dict(globals(), **locals()))
        #        decoder_updates += layer.get_updates_for(None)
        #def xor_list(lst1, lst2): 
        #    return  [value for value in lst1+lst2 if (value not in lst1) or (value not in lst2)  ]  
            
        #decoder_def_updates = self.decoder.updates.copy()
        xr = self.decoder ([c_xa,s_xa])
        xt = self.decoder ([c_xa,s_xb])
        #decoder_updates = xor_list (decoder_def_updates, self.decoder.updates )
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        d_xr, d_xr_feat = self.dis(xr)
        d_xt, d_xt_feat = self.dis(xt)

        d_xa, d_xa_feat = self.dis(xa)
        d_xb, d_xb_feat = self.dis(xb)
        
        def dis_gather(x,l, func=None):            
            tensors = []
            for i in range(self.batch_size):
                t = x[i:i+1,:,:, l[i,0]]
                if func is not None:
                    t = func(t)
                tensors += [t]
            return tensors
            
        def dis_gather_batch_mean(x,l, func=None):
            #h,w,c = K.int_shape(x)[1:]
            x_shape = K.shape(x)
            b,h,w,c = x_shape[0],x_shape[1],x_shape[2],x_shape[3]
            b,h,w,c = [ K.cast(x, K.floatx()) for x in [b,h,w,c] ]
            
            tensors = dis_gather(x,l,func)
            return K.sum(tensors, axis=[1,2,3]) / (h*w)
            
            return K.sum(tensors, axis=[1,2,3]) / (h*w)
            
        def dis_gather_mean(x,l, func=None):
            x_shape = K.shape(x)
            b,h,w,c = x_shape[0],x_shape[1],x_shape[2],x_shape[3]
            b,h,w,c = [ K.cast(x, K.floatx()) for x in [b,h,w,c] ]
            
            tensors = dis_gather(x,l,func)
            return K.sum(tensors) / (b*h*w)
            #return K.sum(tensors) / (h*w*self.batch_size)

        
        d_xr_la = dis_gather_mean(d_xr, la)
        d_xt_lb = dis_gather_mean(d_xt, lb)
        
        d_xb_lb = dis_gather_batch_mean(d_xb, lb)
        
        d_xb_lb_real = dis_gather_mean(d_xb, lb, lambda x: K.relu(1.0-x) )
        d_xt_lb_fake = dis_gather_mean(d_xt, lb, lambda x: K.relu(1.0+x) )
        
        G_c_rec = K.mean(K.abs(K.mean(d_xr_feat, axis=[1,2]) - K.mean(d_xa_feat, axis=[1,2]))) #, axis=1 * 1.0
        G_m_rec = K.mean(K.abs(K.mean(d_xt_feat, axis=[1,2]) - K.mean(d_xb_feat, axis=[1,2]))) #, axis=1 * 1.0
        G_x_rec = 0.1 * K.mean(K.abs(xr-xa)) # , axis=[1,2,3] 0.1 def
        
        G_d_xr_la = -d_xr_la*0.5
        G_d_xt_lb = -d_xt_lb*0.5
        
        G_loss = G_d_xr_la + G_d_xt_lb + G_x_rec + G_c_rec + G_m_rec
        G_weights = self.enc_class_model.trainable_weights + self.enc_content.trainable_weights + self.decoder.trainable_weights
        ######
        
        D_real = d_xb_lb_real #1.0 * 
        D_fake = d_xt_lb_fake #1.0 * 

        l_reg = 10 * K.sum( K.gradients( d_xb_lb, xb )[0] ** 2 ) # , axis=[1,2,3] / self.batch_size )

        D_loss = D_real + D_fake + l_reg

        D_weights = self.dis.trainable_weights

        #we are using K.function API, so layer update ops must be gathered explicitly
 
        #updates_layers = []
        #for layer in self.decoder.layers:
        #    if type(layer) == FUNITAdain:        
        #        import code
        #        code.interact(local=dict(globals(), **locals()))
        #        G_updates += layer.get_updates_for(None)

        self.G_train = K.function ([xa, la, xb, lb],[G_loss], self.G_opt.get_updates(G_loss, G_weights) + decoder_updates )
        self.D_train = K.function ([xa, la, xb, lb],[D_loss], self.D_opt.get_updates(D_loss, D_weights) )
        
        self.get_average_class_code = K.function ([xa],[s_xa_mean])
        
        xr_one = self.decoder ([c_xa,s_xa_one])
        self.G_convert = K.function  ([xa,s_xa_one],[xr_one])

        if not self.is_first_run():
            self.load_weights_safe(self.get_model_filename_list())
        else:
            initer = keras.initializers.he_normal()
            
            CA_conv_weights_list = []
            for model, _ in self.get_model_filename_list():
                if type(model) == keras.models.Model:
                    for layer in model.layers:
                        if type(layer) == keras.layers.Conv2D or type(layer) == keras.layers.Dense:
                            w = layer.weights[0]
                            K.set_value( w, K.get_value(initer(K.int_shape(w)))  )
                        

        #import code
        #code.interact(local=dict(globals(), **locals()))

        t = SampleProcessor.Types
        face_type = t.FACE_TYPE_FULL
        if self.is_training_mode:           

            output_sample_types=[ {'types': (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':resolution, 'normalize_tanh':True} ]

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip),
                        output_sample_types=output_sample_types )
                   ])
        else:
            #self.G_convert = K.function([real_B0],[fake_A0])
            
            generator = SampleGeneratorFace(self.training_data_src_path, batch_size=1,
                                sample_process_options=SampleProcessor.Options(),
                                output_sample_types=[ {'types': (t.IMG_SOURCE, face_type, t.MODE_BGR), 'resolution':resolution, 'normalize_tanh':True} ] )
            
            io.log_info("Calculating average src face style...")
            codes = []
            for i in range(generator.get_total_sample_count()):
                codes += self.get_average_class_code( generator.generate_next() )
                
            self.average_class_code = np.mean ( np.array(codes), axis=0 )[None,...]
            
        self.check = self.decoder.layers[5].weights[-2]

    #override
    def get_model_filename_list(self):
        return [[self.enc_class_model, 'enc_class_model.h5'],
                [self.enc_content,     'enc_content.h5'],
                [self.decoder,         'decoder.h5'],
                [self.dis,             'dis.h5'],
                [self.G_opt,           'G_opt.h5'],
                [self.D_opt,           'D_opt.h5'],
                ]

    #override
    def onSave(self):
        self.save_weights_safe(self.get_model_filename_list())

    def shuffle_gen_samples (self, generators_samples):
        bs = self.batch_size
        lbs = bs // 2
        hbs = bs - lbs

        src, = generators_samples[0]
        dst, = generators_samples[1]

        xa = np.concatenate ( [src[0:lbs], dst[0:lbs]], axis=0 )

        la = np.concatenate ( [ np.array ([0]*lbs, np.int32),
                                np.array ([1]*lbs, np.int32) ] )

        xb = np.concatenate ( [src[lbs:], dst[lbs:]], axis=0 )

        lb = np.concatenate ( [ np.array ([0]*hbs, np.int32),
                                np.array ([1]*hbs, np.int32) ] )

        rnd_list = np.arange(lbs*2)
        np.random.shuffle(rnd_list)
        xa = xa[rnd_list,...]
        la = la[rnd_list,...]
        la = la[...,None]

        rnd_list = np.arange(hbs*2)
        np.random.shuffle(rnd_list)
        xb = xb[rnd_list,...]
        lb = lb[rnd_list,...]
        lb = lb[...,None]
        return xa, la, xb, lb

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        xa,la,xb,lb = self.shuffle_gen_samples(generators_samples)

        D_loss, = self.D_train ([xa,la,xb,lb])
        G_loss, = self.G_train ([xa,la,xb,lb])

        return ( ('G_loss', G_loss), ('D_loss', D_loss), )

    #override
    def onGetPreview(self, generators_samples):
        xa  = generators_samples[0][0]
        xb  = generators_samples[1][0]

        view_samples = min(4, xa.shape[0])


        s_xa_mean = self.get_average_class_code([xa])[0][None,...]
        s_xb_mean = self.get_average_class_code([xb])[0][None,...]

        s_xab_mean = self.get_average_class_code([ np.concatenate( [xa,xb], axis=0) ])[0][None,...]
        
        lines = []

        for i in range(view_samples):
            xarX = np.clip (self.G_convert  ([ xa[i:i+1], s_xa_mean  ] )[0][0] / 2 + 0.5, 0, 1)
            xbrX = np.clip (self.G_convert  ([ xb[i:i+1], s_xb_mean  ] )[0][0] / 2 + 0.5, 0, 1)
            xbtX = np.clip (self.G_convert  ([ xb[i:i+1], s_xa_mean  ] )[0][0] / 2 + 0.5, 0, 1)            

            lines += [ np.concatenate( (xa[i] / 2 + 0.5, xarX, xb[i] / 2 + 0.5, xbrX, xbtX), axis=1) ]

        r = np.concatenate ( lines, axis=0 )
        return [ ('TrueFace', r ) ]

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.G_convert ([ np.zeros ( (1, self.options['resolution'], self.options['resolution'], 3), dtype=np.float32 ), self.average_class_code ])
        else:
            bgr, = self.G_convert ([  face[np.newaxis,...]*2-1, self.average_class_code  ])
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
    def ClassModelEncoderFlow(downs=4, ngf=64, latent_dim=64):
        exec (nnlib.import_all(), locals(), globals())

        def ResBlock(dim, norm_func=None):
            def func(input):
                x = input
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = norm_func()(x)
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = norm_func()(x)

                return Add()([x,input])
            return func

        def func(x):
            x = Conv2D (ngf, kernel_size=7, strides=1, padding='same', activation='relu')(x)

            for i in range(downs):
                x = Conv2D (ngf * min ( 2**3, 2**(i+1) ), kernel_size=4, strides=2, padding='same', activation='relu')(x)

            x = GlobalAveragePooling2D()(x)
            x = Dense(ngf)(x)
            return x

        return func

    @staticmethod
    def ContentEncoderFlow(downs=2, ngf=64, n_res_blks=2):
        exec (nnlib.import_all(), locals(), globals())

        def ResBlock(dim):
            def func(input):
                x = input
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = InstanceNormalization()(x)
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = InstanceNormalization()(x)

                return Add()([x,input])
            return func

        def func(x):
            x = Conv2D (ngf, kernel_size=7, strides=1, padding='same')(x)
            x = InstanceNormalization()(x)
            x = ReLU()(x)

            for i in range(downs):
                x = Conv2D (ngf * 2**(i+1), kernel_size=4, strides=2, padding='same')(x)
                x = InstanceNormalization()(x)
                x = ReLU()(x)

            for i in range(n_res_blks):
                x = ResBlock( ngf * 2**downs )(x)
                
            return x

        return func

    @staticmethod
    def DecoderFlow(ups, n_res_blks=2,  nf_mlp=256, n_mlp_blks=2 ):
        exec (nnlib.import_all(), locals(), globals())

        def ResBlock(dim):
            def func(input):
                inp, mlp = input
                x = inp
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = FUNITAdain()([x,mlp])
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = FUNITAdain()([x,mlp])
                return Add()([x,inp])
            return func

        def func(inputs):
            x , class_code = inputs

            ngf = x.shape[-1].value

            ### MLP block inside decoder
            mlp = class_code
            for i in range(n_mlp_blks):
                mlp = Dense(nf_mlp, activation='relu')(mlp)

            for i in range(n_res_blks):
                x = ResBlock(ngf)( [x,mlp] )

            for i in range(ups):
                
                #x = Conv2D (4* (ngf // 2**(i+1)), kernel_size=3, strides=1, padding='same')(x)
                #x = SubpixelUpscaler()(x)
                
                x = UpSampling2D()(x)
                x = Conv2D (ngf // 2**(i+1), kernel_size=5, strides=1, padding='same')(x)
                x = InstanceNormalization()(x)
                x = ReLU()(x)

            return Conv2D (3, kernel_size=7, strides=1, padding='same', activation='tanh')(x)

        return func


    @staticmethod
    def DiscriminatorFlow(bs, nf, n_res_blks, num_classes ):
        exec (nnlib.import_all(), locals(), globals())

        n_layers = n_res_blks // 2

        def ActFirstResBlock(fout):
            def func(x):
                fin = K.int_shape(x)[-1]
                fhid = min(fin, fout)

                if fin != fout:
                    x_s = Conv2D (fout, kernel_size=1, strides=1, padding='valid', use_bias=False)(x)
                else:
                    x_s = x

                x = LeakyReLU(0.2)(x)
                x = Conv2D (fhid, kernel_size=3, strides=1, padding='same')(x)
                x = LeakyReLU(0.2)(x)
                x = Conv2D (fout, kernel_size=3, strides=1, padding='same')(x)
                return  Add()([x_s, x])

            return func

        def func( x ):
            l_nf = nf
            x = Conv2D (l_nf, kernel_size=7, strides=1, padding='same')(x)
            for i in range(n_layers-1):
                l_nf_out = min( l_nf*2, 1024 )
                x = ActFirstResBlock(l_nf)(x)
                x = ActFirstResBlock(l_nf_out)(x)
                x = AveragePooling2D( pool_size=3, strides=2, padding='same' ) (x)
                l_nf = min( l_nf*2, 1024 )

            l_nf_out = min( l_nf*2, 1024 )
            x        = ActFirstResBlock(l_nf)(x)
            feat = x = ActFirstResBlock(l_nf_out)(x)

            x = LeakyReLU(0.2)(x)
            x = Conv2D (num_classes, kernel_size=1, strides=1, padding='valid')(x)

            return x, feat

        return func


Model = FUNITModel
