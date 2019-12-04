import numpy as np

from facelib import FaceType
from interact import interact as io
from models import ModelBase
from nnlib import nnlib
from samplelib import *



        
class FUNIT(object):
    VERSION = 1
    def __init__ (self, face_type_str,
                        batch_size,
                        encoder_nf=64,
                        encoder_downs=2,
                        encoder_res_blk=2,
                        mlp_blks=2,
                        dis_nf=64,
                        dis_res_blks=10,
                        num_classes=2,
                        subpixel_decoder=True,
                        initialize_weights=True,

                        load_weights_locally=False,
                        weights_file_root=None,

                        is_training=True,
                        tf_cpu_mode=0,
                        ):
        exec( nnlib.import_all(), locals(), globals() )

        self.batch_size = batch_size
        bgr_shape = (128, 128, 3)
        label_shape = (1,)

        self.encoder = modelify ( FUNIT.ContentEncoderFlow(downs=encoder_downs, nf=encoder_nf, n_res_blks=encoder_res_blk) ) ( Input(bgr_shape) )
        self.enc_class_model = modelify ( FUNIT.ClassModelEncoderFlow(downs=4, nf=64, latent_dim=64) ) ( Input(bgr_shape) )
        
        self.decoder     = modelify ( FUNIT.DecoderFlow(ups=encoder_downs, n_res_blks=encoder_res_blk, mlp_blks=mlp_blks, subpixel_decoder=subpixel_decoder  ) ) \
                             ( [ Input(K.int_shape(self.encoder.outputs[0])[1:], name="decoder_input_1"),
                                 Input(K.int_shape(self.enc_class_model.outputs[0])[1:], name="decoder_input_2")
                               ] )

        self.G_opt = RMSprop(lr=5e-5)#, decay=0.0001)#, tf_cpu_mode=tf_cpu_mode)

        warped_src, warped_dst = Input(bgr_shape), Input(bgr_shape)
        target_src, target_dst = Input(bgr_shape), Input(bgr_shape)
                

        s_src_one = Input( (  K.int_shape(self.enc_class_model.outputs[0])[-1],), name="s_xa_input")

        src_code = self.encoder(warped_src)
        dst_code = self.encoder(warped_dst)

        s_src = self.enc_class_model(target_src)
        s_dst = self.enc_class_model(target_dst)
        s_src_mean = K.mean(s_src, axis=0)

        rec_src = self.decoder ([src_code,s_src])
        rec_dst = self.decoder ([dst_code,s_dst])
        rec_src_one = self.decoder ([src_code,s_src_one])

        G_loss =  K.mean ( 10*dssim(kernel_size=int(128/11.6),max_value=2.0)( rec_src+1, target_src+1) )
        G_loss += K.mean ( 10*K.square( rec_src - target_src ) )
        G_loss += K.mean ( 10*dssim(kernel_size=int(128/11.6),max_value=2.0)( rec_dst+1, target_dst+1) )
        G_loss += K.mean ( 10*K.square( rec_dst - target_dst ) )

        G_weights = self.enc_class_model.trainable_weights + self.encoder.trainable_weights + self.decoder.trainable_weights
        ######

        self.G_train = K.function ([warped_src, target_src, warped_dst, target_dst],[G_loss], self.G_opt.get_updates(G_loss, G_weights) )

        self.get_average_class_code = K.function ([target_src],[s_src_mean])
        self.G_convert = K.function  ([warped_src,s_src_one],[rec_src_one])

        #if initialize_weights:
        #    #gather weights from layers for initialization
        #    weights_list = []
        #    for model, _ in self.get_model_filename_list():
        #        if type(model) == keras.models.Model:
        #            for layer in model.layers:
        #                if type(layer) == FUNITAdain:
        #                    weights_list += [ x for x in layer.weights if 'kernel' in x.name ]
        #                elif  type(layer) == keras.layers.Conv2D or type(layer) == keras.layers.Dense:
        #                    weights_list += [ layer.weights[0] ]

        #    initer = keras.initializers.he_normal()
        #    for w in weights_list:
        #        K.set_value( w, K.get_value(initer(K.int_shape(w)))  )

        #if not self.is_first_run():
        #    self.load_weights_safe(self.get_model_filename_list())



        if load_weights_locally:
            pass
        #f weights_file_root is not None:
        #   weights_file_root = Path(weights_file_root)
        #lse:
        #   weights_file_root = Path(__file__).parent
        #elf.weights_path = weights_file_root / ('FUNIT_%s.h5' % (face_type_str) )
        #f load_weights:
        #   self.model.load_weights (str(self.weights_path))



    def get_model_filename_list(self):
        return [[self.encoder,         'encoder.h5'],
                [self.enc_class_model, 'enc_class_model.h5'],
                [self.decoder,         'decoder.h5'],
                [self.G_opt,           'G_opt.h5'],
                ]

    #def save_weights(self):
    #    self.model.save_weights (str(self.weights_path))

    def train(self, warped_src, target_src, warped_dst, target_dst):
        G_loss, = self.G_train ([warped_src, target_src, warped_dst, target_dst])
        return G_loss
        
    def get_average_class_code(self, *args, **kwargs):
        return self.get_average_class_code(*args, **kwargs)
        
    def convert(self, *args, **kwargs):
        return self.G_convert(*args, **kwargs)

    @staticmethod
    def ContentEncoderFlow(downs=2, nf=64, n_res_blks=2):
        exec (nnlib.import_all(), locals(), globals())
        
        def downscale (dim, kernel_size=5, dilation_rate=1, use_activator=True):
            def func(x):
                if not use_activator:
                    return Conv2D(dim, kernel_size=kernel_size, strides=2, dilation_rate=dilation_rate, padding='same')(x)
                else:
                    return LeakyReLU(0.1)(Conv2D(dim, kernel_size=kernel_size, strides=2, dilation_rate=dilation_rate, padding='same')(x))
            return func
            
        def upscale (dim, size=(2,2)):
            def func(x):
                return SubpixelUpscaler(size=size)(LeakyReLU(0.1)(Conv2D(dim * np.prod(size) , kernel_size=3, strides=1, padding='same')(x)))
            return func
                
        def ResBlock(dim):
            def func(input):
                x = input
                x = Conv2D(dim, 3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                x = InstanceNormalization()(x)
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                x = InstanceNormalization()(x)

                return Add()([x,input])
            return func

        def func(x):
            
            x = downscale(nf  , 5, 1 )(x)
            x = downscale(nf*2, 5, 1 )(x)
            x = downscale(nf*4, 5, 1 )(x)
            x = downscale(nf*8, 5, 1 )(x)
                        
            x = Dense(256)(Flatten()(x))
            x = Dense(8 * 8 * 256)(x)
            x = Reshape((8, 8, 256))(x)
            
            return x

        return func

    @staticmethod
    def ClassModelEncoderFlow(downs=4, nf=64, latent_dim=64):
        exec (nnlib.import_all(), locals(), globals())

        def func(x):
            x = Conv2D (nf, kernel_size=7, strides=1, padding='valid', activation='relu')(ZeroPadding2D(3)(x))
            for i in range(downs):
                x = Conv2D (nf * min ( 4, 2**(i+1) ), kernel_size=4, strides=2, padding='valid', activation='relu')(ZeroPadding2D(1)(x))
            x = GlobalAveragePooling2D()(x)
            x = Dense(nf)(x)
            return x

        return func

    @staticmethod
    def DecoderFlow(ups, n_res_blks=2, mlp_blks=2, subpixel_decoder=False ):
        exec (nnlib.import_all(), locals(), globals())


        def upscale (self, dim, size=(2,2)):
            def func(x):
                return SubpixelUpscaler(size=size)(LeakyReLU(0.1)(Conv2D(dim * np.prod(size) , kernel_size=3, strides=1, padding='same')(x)))
            return func
            
        def ResBlock(dim):
            def func(input):
                inp, mlp = input
                x = inp
                x = Conv2D(dim, 3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                x = FUNITAdain(kernel_initializer='he_normal')([x,mlp])
                x = ReLU()(x)
                x = Conv2D(dim, 3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                x = FUNITAdain(kernel_initializer='he_normal')([x,mlp])
                return Add()([x,inp])
            return func

        def func(inputs):
            x , class_code = inputs
            
            nf = K.int_shape(x)[-1]
            
            ### MLP block inside decoder            
            mlp = Lambda ( lambda x: K.tile(x, (1,nf) ) ) (class_code)
            for i in range(mlp_blks):
                mlp = Dense(nf, activation='relu')(mlp)
            
            x = FUNITAdain(kernel_initializer='he_normal')([x,mlp])
            x = InstanceNormalization()(x)
            x = ReLU()(x)
            
            x_mlp = Dense(8* 8 * 256)(mlp)
            x_mlp = Reshape((8, 8, 256))(x_mlp)
            x_mlp = InstanceNormalization()(x_mlp)
            x_mlp = ReLU()(x_mlp)
            
                
            x = Concatenate()([x,x_mlp])
            
            for i in range(4):
                d = nf# // 2**(i+1)
                x = Conv2D (4*d, kernel_size=3, strides=1, padding='same')(x)
                x = SubpixelUpscaler()(x)
                #x = InstanceNormalization()(x)
                x = ReLU()(x)                
                #x = ResBlock(d)( [x,mlp] )
                
                  

            return Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)

        return func



    @staticmethod
    def DiscriminatorFlow(nf, n_res_blks, num_classes ):
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
                x = Conv2D (fhid, kernel_size=3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                x = LeakyReLU(0.2)(x)
                x = Conv2D (fout, kernel_size=3, strides=1, padding='valid')(ZeroPadding2D(1)(x))
                return  Add()([x_s, x])

            return func

        def func( x ):
            l_nf = nf
            x = Conv2D (l_nf, kernel_size=7, strides=1, padding='valid')(ZeroPadding2D(3)(x))
            for i in range(n_layers-1):
                l_nf_out = min( l_nf*2, 1024 )
                x = ActFirstResBlock(l_nf)(x)
                x = ActFirstResBlock(l_nf_out)(x)
                x = AveragePooling2D( pool_size=3, strides=2, padding='valid' )(ZeroPadding2D(1)(x))
                l_nf = min( l_nf*2, 1024 )

            l_nf_out = min( l_nf*2, 1024 )
            x        = ActFirstResBlock(l_nf)(x)
            feat = x = ActFirstResBlock(l_nf_out)(x)

            x = LeakyReLU(0.2)(x)
            x = Conv2D (num_classes, kernel_size=1, strides=1, padding='valid')(x)

            return x, feat

        return func
        
class TrueFaceModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_resolution = 128
        default_face_type = 'f'

        if is_first_run:
            resolution = self.options['resolution'] = io.input_int(f"分辨率( 64-256 帮助 :? 跳过:{default_resolution}) : ", default_resolution, help_message="更高的分辨率需要更多的VRAM和训练时间。 数值调整成16的倍数。")
            resolution = np.clip (resolution, 64, 256)
            while np.modf(resolution / 16)[0] != 0.0:
                resolution -= 1
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)

        if is_first_run:
            self.options['face_type'] = io.input_str ("半脸（h）全脸（f）? (帮助:? 跳过:f) : ", default_face_type, ['h','f'], help_message="").lower()
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            
        if (is_first_run or ask_override) and 'tensorflow' in self.device_config.backend:
            def_optimizer_mode = self.options.get('optimizer_mode', 3)
            self.options['optimizer_mode'] = io.input_int ("优化模式? ( 1,2,3 帮助:? 跳过:%d) : " % (def_optimizer_mode), def_optimizer_mode, help_message="1 - 没有变化。2 - 允许您训练x2更大的网络消耗内存。3 - 允许你训练x3更大的网络消耗大量的内存和更慢，取决于CPU的功率。")
        else:
            self.options['optimizer_mode'] = self.options.get('optimizer_mode', 1)

    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({2:1,3:1,4:4})

        resolution = self.options['resolution']
        face_type = self.face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        self.model = FUNIT( face_type_str=FaceType.toString(face_type),
                            batch_size=self.batch_size,
                            encoder_nf=64,
                            encoder_downs=3,
                            encoder_res_blk=2,
                            mlp_blks=2,
                            dis_nf=64,
                            dis_res_blks=8,
                            num_classes=2,
                            subpixel_decoder=True,
                            initialize_weights=self.is_first_run(),
                            is_training=self.is_training_mode,
                            tf_cpu_mode=self.options['optimizer_mode']-1
                           )

        if not self.is_first_run():
            self.load_weights_safe(self.model.get_model_filename_list())

        t = SampleProcessor.Types
        face_type = t.FACE_TYPE_FULL if self.options['face_type'] == 'f' else t.FACE_TYPE_HALF
        if self.is_training_mode:

            output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_GGG), 'resolution':resolution, 'normalize_tanh':True},
                                    {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution': resolution, 'normalize_tanh':True},
                                    {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution } ]
                         

            self.set_training_data_generators ([
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True),
                        output_sample_types=output_sample_types ),

                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=True),
                        output_sample_types=output_sample_types )
                   ])


    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list()

    #override
    def onSave(self):
        self.save_weights_safe(self.model.get_model_filename_list())

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, target_src, target_srcm = generators_samples[0]
        warped_dst, target_dst, target_dstm = generators_samples[1]
        
        G_loss = self.model.train(warped_src, target_src, warped_dst, target_dst )
        
        return ( ('G_loss', G_loss), )

    #override
    def onGetPreview(self, generators_samples):
        xa  = generators_samples[0][0]
        xb  = generators_samples[1][0]

        view_samples = min(4, xa.shape[0])

        lines = []
        
        s_xa_mean = self.model.get_average_class_code([xa])[0][None,...]
        s_xb_mean = self.model.get_average_class_code([xb])[0][None,...]

        s_xab_mean = self.model.get_average_class_code([ np.concatenate( [xa,xb], axis=0) ])[0][None,...]


        for i in range(view_samples):
            xaxa, = self.model.convert  ([ xa[i:i+1], s_xa_mean  ] )
            xbxb, = self.model.convert  ([ xb[i:i+1], s_xb_mean  ] )
            xbxa, = self.model.convert  ([ xb[i:i+1], s_xab_mean  ] )

            xa_i,xb_i,xaxa,xbxb,xbxa = [ np.clip(x/2+0.5, 0, 1) for x in [xa[i], xb[i], xaxa[0],xbxb[0],xbxa[0]] ]

            lines += [ np.concatenate( (xa_i, xaxa, xb_i, xbxb, xbxa), axis=1) ]

        r = np.concatenate ( lines, axis=0 )
        return [ ('TrueFace', r ) ]

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.model.convert ([ np.zeros ( (1, self.options['resolution'], self.options['resolution'], 3), dtype=np.float32 ), self.average_class_code ])
        else:
            bgr, = self.model.convert ([  face[np.newaxis,...]*2-1, self.average_class_code  ])
            return bgr[0] / 2 + 0.5

    #override
    def get_ConverterConfig(self):

        import converters
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), converters.ConverterConfigMasked(face_type=self.face_type,
                                     default_mode = 1,
                                     clip_hborder_mask_per=0.0625 if (self.face_type == FaceType.FULL) else 0,
                                    )

Model = TrueFaceModel
