from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import nnlib
from samplelib import *



class FUNITModel(ModelBase):

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

        person_id_max_count = SampleGeneratorFace.get_person_id_max_count(self.training_data_src_path)

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

        self.dis = modelify ( FUNITModel.DiscriminatorFlow(self.batch_size, nf=ndf, n_res_blks=10, num_classes=person_id_max_count ) ) (Input(bgr_shape))

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

        xr = self.decoder ([c_xa,s_xa])
        xt = self.decoder ([c_xa,s_xb])

        xr_one = self.decoder ([c_xa,s_xa_one])


        d_xr, d_xr_feat = self.dis(xr)
        d_xt, d_xt_feat = self.dis(xt)

        d_xa, d_xa_feat = self.dis(xa)
        d_xb, d_xb_feat = self.dis(xb)

        def dis_gather(x,l):
            h,w,c = K.int_shape(x)[1:]
            out = []
            for i in range(self.batch_size):
                out += [  x[i:i+1,:,:, l[i,0]] ]

            return K.sum(out, axis=[1,2,3]) / (h*w)

        d_xr_la = dis_gather(d_xr, la)
        d_xt_lb = dis_gather(d_xt, lb)
        d_xa_la = dis_gather(d_xa, la)
        d_xb_lb = dis_gather(d_xb, lb)

        G_c_rec = K.mean(K.abs(K.mean(d_xr_feat, axis=[1,2]) - K.mean(d_xa_feat, axis=[1,2])), axis=1) # * 1.0
        G_m_rec = K.mean(K.abs(K.mean(d_xt_feat, axis=[1,2]) - K.mean(d_xb_feat, axis=[1,2])), axis=1) # * 1.0
        G_x_rec = 0.1 * K.mean(K.abs(xr-xa), axis=[1,2,3]) #0.1 def

        G_d_xr_la = -d_xr_la*0.5
        G_d_xt_lb = -d_xt_lb*0.5



        #l_adv = 0.5* ( -K.mean(d_xr_la) - K.mean(d_xt_lb) )
        #G_loss = 1 * l_adv + G_x_rec + (G_c_rec+G_m_rec)
        G_loss = G_d_xr_la + G_d_xt_lb + G_x_rec + G_c_rec + G_m_rec
        G_weights = self.enc_class_model.trainable_weights + self.enc_content.trainable_weights + self.decoder.trainable_weights
        ######



        D_real = K.relu(1.0 - d_xb_lb) #1.0 *
        D_fake = K.relu(1.0 + d_xt_lb) #1.0 *

        l_reg = 10 * K.sum( K.gradients( d_xb_lb, xb )[0] ** 2, axis=[1,2,3] ) #/ self.batch_size )

        D_loss = D_real + D_fake + l_reg

        D_weights = self.dis.trainable_weights


        self.G_train = K.function ([xa, la, xb, lb],[K.mean(G_loss)], self.G_opt.get_updates(G_loss, G_weights) )
        self.D_train = K.function ([xa, la, xb, lb],[K.mean(D_loss)], self.D_opt.get_updates(D_loss, D_weights) )

        self.get_average_class_code = K.function ([xa],[s_xa_mean])
        self.G_convert = K.function  ([xa,s_xa_one],[xr_one])

        if not self.is_first_run():
            self.load_weights_safe(self.get_model_filename_list())
        else:
            initer = keras.initializers.glorot_normal()
            for model, _ in self.get_model_filename_list():
                if type(model) == keras.models.Model:
                    for layer in model.layers:
                        if type(layer) == keras.layers.Conv2D or type(layer) == keras.layers.Dense:
                            w = layer.weights[0]
                            K.set_value( w, K.get_value(initer(K.int_shape(w)))  )

        if self.is_training_mode:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            output_sample_types=[ {'types': (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':128, 'normalize_tanh':True} ]

            self.set_training_data_generators ([
                        SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),

                        SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),

                        SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),

                        SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types, person_id_mode=True ),
                    ])

        else:
            #self.G_convert = K.function([real_B0],[fake_A0])
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            generator = SampleGeneratorFace(self.training_data_src_path, batch_size=1,
                                sample_process_options=SampleProcessor.Options(),
                                output_sample_types=[ {'types': (t.IMG_SOURCE, face_type, t.MODE_BGR), 'resolution':resolution, 'normalize_tanh':True} ] )

            io.log_info("Calculating average src face style...")
            codes = []
            for i in range(generator.get_total_sample_count()):
                codes += self.get_average_class_code( generator.generate_next() )

            self.average_class_code = np.mean ( np.array(codes), axis=0 )[None,...]



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

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        xa,la = generators_samples[0]
        xb,lb = generators_samples[1]

        D_loss, = self.D_train ([xa,la,xb,lb])
        G_loss, = self.G_train ([xa,la,xb,lb])

        return ( ('G_loss', G_loss), ('D_loss', D_loss), )

    #override
    def onGetPreview(self, generators_samples):
        xa  = generators_samples[0][0]
        xb  = generators_samples[1][0]
        ta  = generators_samples[2][0]
        tb  = generators_samples[3][0]

        view_samples = min(4, xa.shape[0])

        lines_train = []
        lines_test = []

        for i in range(view_samples):

            s_xa = self.get_average_class_code([ xa[i:i+1] ])[0][None,...]
            s_xb = self.get_average_class_code([ xb[i:i+1] ])[0][None,...]

            s_ta = self.get_average_class_code([ ta[i:i+1] ])[0][None,...]
            s_tb = self.get_average_class_code([ tb[i:i+1] ])[0][None,...]

            xaxa = self.G_convert  ([ xa[i:i+1], s_xa  ] )[0][0]
            xbxb = self.G_convert  ([ xb[i:i+1], s_xb  ] )[0][0]
            xaxb = self.G_convert  ([ xa[i:i+1], s_xb  ] )[0][0]
            xbxa = self.G_convert  ([ xb[i:i+1], s_xa  ] )[0][0]

            tata = self.G_convert  ([ ta[i:i+1], s_ta  ] )[0][0]
            tbtb = self.G_convert  ([ tb[i:i+1], s_tb  ] )[0][0]
            tatb = self.G_convert  ([ ta[i:i+1], s_tb  ] )[0][0]
            tbta = self.G_convert  ([ tb[i:i+1], s_ta  ] )[0][0]

            line_train = [ xa[i], xaxa, xb[i], xbxb, xaxb, xbxa ]
            line_test =  [ ta[i], tata, tb[i], tbtb, tatb, tbta ]

            lines_train += [ np.concatenate([ np.clip(x/2+0.5,0,1) for x in line_train], axis=1) ]
            lines_test  += [ np.concatenate([ np.clip(x/2+0.5,0,1) for x in line_test ], axis=1) ]

        lines_train = np.concatenate ( lines_train, axis=0 )
        lines_test = np.concatenate ( lines_test, axis=0 )
        return [ ('TRAIN', lines_train ), ('TEST', lines_test) ]

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
            x = Lambda( lambda x: x[:,None,None,:], output_shape=(1,1,K.int_shape(x)[-1]) ) (x)
            x = Conv2D (ngf, kernel_size=1, strides=1, padding='valid')(x)
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

            res_ngf = K.int_shape(x)[-1]
            for i in range(n_res_blks):
                x = ResBlock(res_ngf)(x)


            return x

        return func

    @staticmethod
    def DecoderFlow(ups, n_res_blks=2,  nf_mlp=256, n_mlp_blks=2 ):
        exec (nnlib.import_all(), locals(), globals())


        class AdaptiveInstanceNormalization2D(KL.Layer):
            """
            my own impl of adain.
            differents from NVLabs/FUNIT:
            1) I moved two dense blocks inside this layer,
               so we don't need to slice outter MLP block, just pass it inside.
               also size of dense blocks calculated automatically
            2) I don't use running_mean and running_var, because they are part of batchnormalization,
               but InstanceNormalization doesn't use them

            """
            def __init__(self, axis=-1, epsilon=1e-3, **kwargs):
                self.axis = axis
                self.epsilon = epsilon
                super(AdaptiveInstanceNormalization2D, self).__init__(**kwargs)

            def build(self, input_shape):
                self.input_spec = None
                x, mlp = input_shape
                input_dim = units = x[-1]

                self.kernel1 = self.add_weight(shape=(input_dim, units), initializer='glorot_normal', name='kernel1')
                self.bias1 = self.add_weight(shape=(units,), initializer='zeros', name='bias1')
                self.kernel2 = self.add_weight(shape=(input_dim, units), initializer='glorot_normal', name='kernel2')
                self.bias2 = self.add_weight(shape=(units,), initializer='zeros', name='bias2')


                self.built = True

            def call(self, inputs, training=None):
                x, mlp = inputs

                gamma = K.dot(mlp, self.kernel1)
                gamma = K.bias_add(gamma, self.bias1, data_format='channels_last')

                beta = K.dot(mlp, self.kernel2)
                beta = K.bias_add(beta, self.bias2, data_format='channels_last')

                input_shape = K.int_shape(x)

                reduction_axes = list(range(len(input_shape)))
                del reduction_axes[self.axis]
                del reduction_axes[0]

                broadcast_shape = [1] * len(input_shape)
                broadcast_shape[self.axis] = input_shape[self.axis]

                mean = K.mean(x, reduction_axes, keepdims=True)
                stddev = K.std(x, reduction_axes, keepdims=True) + self.epsilon
                normed = (x - mean) / stddev

                normed *= K.reshape(gamma,[-1]+broadcast_shape[1:] )
                normed += K.reshape(beta, [-1]+broadcast_shape[1:] )

                return normed

            def get_config(self):
                config = {'axis': self.axis, 'epsilon': self.epsilon }

                base_config = super(AdaptiveInstanceNormalization2D, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))

            def compute_output_shape(self, input_shape):
                return input_shape

        def ResBlock(dim):
            def func(input):
                inp, mlp = input
                x = inp
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = AdaptiveInstanceNormalization2D()([x,mlp])
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='same')(x)
                x = AdaptiveInstanceNormalization2D()([x,mlp])

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
