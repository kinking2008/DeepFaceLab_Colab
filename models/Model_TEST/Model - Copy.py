from functools import partial

import numpy as np

import mathlib
from facelib import FaceType
from interact import interact as io
from models import ModelBase
from nnlib import nnlib
from samplelib import *


#SAE - Styled AutoEncoder
class SAEv2Model(ModelBase):

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        yn_str = {True:'y',False:'n'}

        default_resolution = 128
        default_archi = 'df'
        default_face_type = 'f'
        default_learn_mask = False

        if is_first_run:
            resolution = io.input_int("分辨率 ( 64-256 帮助:? 跳过:128) : ", default_resolution, help_message="更高的分辨率需要更多的VRAM和训练时间。 数值调整成16的倍数。")
            resolution = np.clip (resolution, 64, 256)
            while np.modf(resolution / 16)[0] != 0.0:
                resolution -= 1
            self.options['resolution'] = resolution

            self.options['face_type'] = io.input_str ("半脸（h）,中脸（mf）,全脸（f）? (帮助:? 跳过:f): ", default_face_type, ['h','mf','f'], help_message="半脸具有更好的分辨率，但脸颊面积较小。中脸比半脸宽30％。").lower()
            self.options['learn_mask'] = io.input_bool ( f"学习面具? (y/n, 帮助:? 跳过:{yn_str[default_learn_mask]} ) : " , default_learn_mask, help_message="学习面具可以帮助模型识别面部方向。不使用学习面具可以减少模型大小，在这种情况下，转换器被迫使用不像预测的那样平滑的“未预测的掩模”。具有样式值的模型可以在没有蒙版的情况下学习并产生相同质量的结果。")
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['face_type'] = self.options.get('face_type', default_face_type)
            self.options['learn_mask'] = self.options.get('learn_mask', default_learn_mask)

        if (is_first_run or ask_override) and 'tensorflow' in self.device_config.backend:
            def_optimizer_mode = self.options.get('optimizer_mode', 1)
            self.options['optimizer_mode'] = io.input_int ("优化模式? ( 1,2,3 帮助:? 跳过:%d) : " % (def_optimizer_mode), def_optimizer_mode, help_message="1 - 没有变化。2 - 允许您训练x2更大的网络消耗内存。3 - 允许你训练x3更大的网络消耗大量的内存和更慢，取决于CPU的功率。")
        else:
            self.options['optimizer_mode'] = self.options.get('optimizer_mode', 1)

        if is_first_run:
            self.options['archi'] = io.input_str ("AE架构 (df, liae 帮助:? 跳过:%s) : " % (default_archi) , default_archi, ['df','liae'], help_message="'df'让面孔更自然。'liae'可以修复过于不同的脸型。").lower() #-s version is slower, but has decreased change to collapse.
        else:
            self.options['archi'] = self.options.get('archi', default_archi)

        default_ae_dims = 256 if 'liae' in self.options['archi'] else 512
        default_e_ch_dims = 21
        default_d_ch_dims = 42
        def_ca_weights = True

        if is_first_run:
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder变暗 (32-1024 帮助:? 跳过:%d) : " % (default_ae_dims) , default_ae_dims, help_message="所有面部信息都将打包到AE暗淡。如果AE暗淡的数量不够，那么例如闭合的眼睛将无法识别。更多的暗淡更好，但需要更多的VRAM。您可以微调模型大小以适合您的GPU。" ), 32, 1024 )
            self.options['e_ch_dims'] = np.clip ( io.input_int("每个通道的编码器变暗 (21-85 帮助:? 跳过:%d) : " % (default_e_ch_dims) , default_e_ch_dims, help_message="更多编码器变暗有助于识别更多面部特征，但需要更多VRAM。您可以微调模型尺寸以适合您的GPU。" ), 21, 85 )
            #default_d_ch_dims = self.options['e_ch_dims']
            self.options['d_ch_dims'] = np.clip ( io.input_int("每个通道的解码器变暗 (10-85 帮助:? 跳过:%d) : " % (default_d_ch_dims) , default_d_ch_dims, help_message="更多解码器变暗有助于获得更好的细节，但需要更多VRAM。您可以微调模型大小以适应您的GPU。" ), 10, 85 )
            self.options['ca_weights'] = io.input_bool (f"使用CA权重? (y/n, 帮助:? 跳过:{yn_str[def_ca_weights]} ) : ", def_ca_weights, help_message="使用“Convolution Aware”权重初始化网络。这可能有助于实现更高精度的模型，但在首次运行时会消耗时间。")
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['e_ch_dims'] = self.options.get('e_ch_dims', default_e_ch_dims)
            self.options['d_ch_dims'] = self.options.get('d_ch_dims', default_d_ch_dims)
            self.options['ca_weights'] = self.options.get('ca_weights', def_ca_weights)

        default_face_style_power = 0.0
        default_bg_style_power = 0.0
        if is_first_run or ask_override:
            def_pixel_loss = self.options.get('pixel_loss', False)
            self.options['pixel_loss'] = io.input_bool (f"使用像素丢失? (y/n, 帮助:? 跳过:{yn_str[def_pixel_loss]} ) : ", def_pixel_loss, help_message="像素丢失可能有助于增强细节和稳定面部颜色。只有在质量不随时间改善的情况下使用它。过早启用此选项会增加模型崩溃的可能性。")

            default_face_style_power = default_face_style_power if is_first_run else self.options.get('face_style_power', default_face_style_power)
            self.options['face_style_power'] = np.clip ( io.input_number("面部风格样式 ( 0.0 .. 100.0 帮助:? 跳过:%.2f) : " % (default_face_style_power), default_face_style_power,
                                                                               help_message="学会转移面部样式细节，例如光线和颜色条件。警告：只有在10k iters之后启用它，当预测的面部足够清晰以开始学习风格时。从0.1值开始并检查历史记录更改。启用此选项会增加 模特崩溃。"), 0.0, 100.0 )

            default_bg_style_power = default_bg_style_power if is_first_run else self.options.get('bg_style_power', default_bg_style_power)
            self.options['bg_style_power'] = np.clip ( io.input_number("背景风格样式( 0.0 .. 100.0 帮助:? 跳过:%.2f) : " % (default_bg_style_power), default_bg_style_power,
                                                                               help_message="学会围绕脸部传递图像。这可以让脸部更像dst。启用此选项会增加模型崩溃的机会。"), 0.0, 100.0 )

            default_apply_random_ct = False if is_first_run else self.options.get('apply_random_ct', False)
            self.options['apply_random_ct'] = io.input_bool (f"将随机颜色转移应用于src 脸部上? (y/n, 帮助:? 跳过:{yn_str[default_apply_random_ct]}) : ", default_apply_random_ct, help_message="通过应用随机dst样本的LCT颜色转移来增加src样本的变化。就像【面部风格样式】学习，但更精确的颜色转移，没有模型崩溃的风险，也不需要额外的GPU资源，但培训时间可能 更长，因为src的面貌变得更加多样化。")

            default_true_face_training = False if is_first_run else self.options.get('true_face_training', False)
            self.options['true_face_training'] = io.input_bool (f"启用 '真脸' 训练? (y/n, 帮助:? 跳过:{yn_str[default_true_face_training]}) : ", default_true_face_training, help_message="使用结果脸将更像src。只有当脸部足够锐利时，才需要在100k次迭代后启用它")


            if nnlib.device.backend != 'plaidML': # todo https://github.com/plaidml/plaidml/issues/301
                default_clipgrad = False if is_first_run else self.options.get('clipgrad', False)
                self.options['clipgrad'] = io.input_bool (f"启用渐变剪辑? (y/n, 帮助:? 跳过:{yn_str[default_clipgrad]}) : ", default_clipgrad, help_message="渐变裁剪减少了模型崩溃的可能性，牺牲了训练速度.")
            else:
                self.options['clipgrad'] = False

        else:
            self.options['pixel_loss'] = self.options.get('pixel_loss', False)
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
            self.options['apply_random_ct'] = self.options.get('apply_random_ct', False)
            self.options['clipgrad'] = self.options.get('clipgrad', False)

        if is_first_run:
            self.options['pretrain'] = io.input_bool ("预训练? (y/n, 帮助:? 跳过:n) : ", False, help_message="通过加载pretrain_CelebA目录下人脸进行预训练。这种技术可能有助于训练具有过度不同的面部形状和src / dst数据的光照条件的伪造生成。用这种方式脸部看起来会有些变形，为了减少这种变形效果，一些模型文件将被初始化但在预训练后不会更新：LIAE：inter_AB.h5 DF：encoder.h5。预训练模型的时间越长，变形的脸就会越多。 然后，保存并再次运行模型就会进入常规模式。")
        else:
            self.options['pretrain'] = False

    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements({6:8})

        resolution = self.options['resolution']
        learn_mask = self.options['learn_mask']

        ae_dims = 64#self.options['ae_dims']
        e_ch_dims = self.options['e_ch_dims']
        d_ch_dims = self.options['d_ch_dims']
        self.pretrain = self.options['pretrain'] = self.options.get('pretrain', False)
        if not self.pretrain:
            self.options.pop('pretrain')

        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        apply_random_ct = self.options.get('apply_random_ct', False)
        self.true_face_training = True#self.options.get('true_face_training', False)
        masked_training = False

        class SAEDFModel(object):
            def __init__(self, batch_size, resolution, ae_dims, e_ch_dims, d_ch_dims, learn_mask):
                super().__init__()
                self.batch_size = batch_size
                self.learn_mask = learn_mask

                output_nc = 3
                bgr_shape = (resolution, resolution, output_nc)
                mask_shape = (resolution, resolution, 1)
                lowest_dense_res = resolution // 16
                e_dims = output_nc*e_ch_dims

                def downscale (dim, kernel_size=5, dilation_rate=1, use_activator=True):
                    def func(x):
                        if not use_activator:
                            return SubpixelDownscaler()(Conv2D(dim // 4, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x))
                        else:
                            return SubpixelDownscaler()(LeakyReLU(0.1)(Conv2D(dim // 4, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x)))
                    return func

                def upscale (dim, size=(2,2)):
                    def func(x):
                        return SubpixelUpscaler(size=size)(LeakyReLU(0.1)(Conv2D(dim * np.prod(size) , kernel_size=3, strides=1, padding='same')(x)))
                    return func
                
                def ResidualBlock(dim):
                    def func(inp):
                        x = Conv2D(dim, kernel_size=3, padding='same')(inp)
                        x = LeakyReLU(0.2)(x)
                        x = Conv2D(dim, kernel_size=3, padding='same')(x)
                        x = Add()([x, inp])
                        x = LeakyReLU(0.2)(x)
                        return x
                    return func
                        
                def enc_flow(e_ch_dims, ae_dims, lowest_dense_res):
                    dims = output_nc * e_ch_dims
                    if dims % 2 != 0:
                        dims += 1

                    def func(inp):                        
                        x = downscale(dims  , 3, 1 )(inp)
                        x = downscale(dims*2, 3, 1 )(x)
                        x = downscale(dims*4, 3, 1 )(x)
                        x0 = downscale(dims*8, 3, 1 )(x)

                        x = downscale(dims  , 5, 1 )(inp)
                        x = downscale(dims*2, 5, 1 )(x)
                        x = downscale(dims*4, 5, 1 )(x)
                        x1 = downscale(dims*8, 5, 1 )(x)

                        x = downscale(dims  , 5, 2 )(inp)
                        x = downscale(dims*2, 5, 2 )(x)
                        x = downscale(dims*4, 5, 2 )(x)
                        x2 = downscale(dims*8, 5, 2 )(x)

                        x = downscale(dims  , 7, 2 )(inp)
                        x = downscale(dims*2, 7, 2 )(x)
                        x = downscale(dims*4, 7, 2 )(x)
                        x3 = downscale(dims*8, 7, 2 )(x)

                        x = Concatenate()([x0,x1,x2,x3])
                        x = Flatten()(x)
                        
                        mean = Dense(ae_dims)(x)
                        logvar = Dense(ae_dims)(x)

                        x = Lambda ( lambda x: x[0] + K.random_normal( K.shape(x[0]) ) * K.exp(x[1]/2)      , output_shape=(ae_dims,) ) ([mean, logvar])
                        return x
                    return func

                def dec_flow(output_nc, d_ch_dims, is_mask=False):
                    dims = output_nc * d_ch_dims
                    if dims % 2 != 0:
                        dims += 1

                    def ResidualBlock(dim):
                        def func(inp):
                            x = Conv2D(dim, kernel_size=3, padding='same')(inp)
                            x = LeakyReLU(0.2)(x)
                            x = Conv2D(dim, kernel_size=3, padding='same')(x)
                            x = Add()([x, inp])
                            x = LeakyReLU(0.2)(x)
                            return x
                        return func

                    def func(x):                               
                        x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
                        x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
                        x = upscale(ae_dims)(x)

                        for i in ( [8,4,2] if not is_mask else [4,2,1] ):
                            x = upscale(dims*i)(x)
                            #if not is_mask:
                            #    x0 = x
                            #    x = upscale( (dims*i)//2 )(x)
                            #    x = ResidualBlock( (dims*i)//2 )(x)
                            #    x = downscale( dims*i, use_activator=False ) (x)
                            #    x = Add()([x, x0])
                            #    x = LeakyReLU(0.2)(x)
  
                        return Conv2D(output_nc, kernel_size=5, padding='same', activation='sigmoid')(x)

                    return func

                def pyr_flow():
                    def func(latent):
                        x = latent

                        x = Dense(1024, activation='relu')(x)
                        x = Dropout(0.5)(x)
                        x = Dense(1024, activation='relu')(x)
                        x = Dense(3, activation='tanh')(x)                            
                        return x
                    return func  
                    
                def id_flow():
                    def func(latent):
                        x = latent

                        x = Dense(1024, activation='relu')(x)
                        x = Dropout(0.5)(x)
                        x = Dense(1024, activation='relu')(x)
                        x = Dense(2, activation='sigmoid')(x)                            
                        return x
                        
                    return func 
                                      
                self.encoder = modelify(enc_flow(e_ch_dims, ae_dims, lowest_dense_res)) ( Input(bgr_shape) )

                #sh = K.int_shape( self.encoder.outputs[2] )[1:]
                sh = (ae_dims+2+3,)

                self.decoder = modelify(dec_flow(output_nc, d_ch_dims)) ( Input(sh) )
                
                sh = (ae_dims,)
                self.pyrest  = modelify(pyr_flow()) ( Input(sh) )

                sh = (ae_dims,)
                self.idest  = modelify(id_flow()) ( Input(sh) )
                
                if learn_mask:
                    self.decoderm = modelify(dec_flow(1, d_ch_dims, is_mask=True)) ( Input(sh) )

                self.src_dst_all_trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights + self.idest.trainable_weights
                self.enc_dec_trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights #+ self.idest.trainable_weights
                self.decoder_trainable_weights = self.decoder.trainable_weights
                
                self.idest_trainable_weights = self.idest.trainable_weights
                self.pyrest_trainable_weights = self.pyrest.trainable_weights
                
                if learn_mask:
                    self.src_dst_mask_trainable_weights = self.encoder.trainable_weights + self.decoderm.trainable_weights

                self.warped_src, self.warped_dst = Input(bgr_shape, name="warped_src"), Input(bgr_shape, name="warped_dst")
                #self.warped_srcm, self.warped_dstm = Input(bgr_shape), Input(bgr_shape)
                self.target_src, self.target_dst = Input(bgr_shape, name="target_src"), Input(bgr_shape, name="target_dst")
                self.target_srcm, self.target_dstm = Input(mask_shape, name="target_srcm"), Input(mask_shape, name="target_dstm")
                
                self.src_pyr = Input ( (3,), name="src_pyr" )
                self.dst_pyr = Input ( (3,), name="dst_pyr" )
                
                #src = [ self.warped_src*self.warped_srcm, self.warped_src*(1-self.warped_srcm) ]
                #dst = [ self.warped_dst*self.warped_dstm, self.warped_dst*(1-self.warped_dstm) ]

                self.src_code = self.encoder( self.warped_src)#*self.warped_srcm )
                self.dst_code = self.encoder( self.warped_dst)#*self.warped_dstm )
                
                self.src_est_pyr = self.pyrest (self.src_code)
                self.dst_est_pyr = self.pyrest (self.dst_code)
                
                self.src_est_id = self.idest (self.src_code)
                self.dst_est_id = self.idest (self.dst_code)
                
                #self.src_mean, self.src_logvar, 
                #self.dst_mean, self.dst_logvar, 
                
                self.src_id = K.tile(K.constant ( [[1,0]], dtype=K.floatx() ), (K.shape(self.dst_code)[0] , 1))
                self.dst_id = K.tile(K.constant ( [[0,1]], dtype=K.floatx() ), (K.shape(self.dst_code)[0] , 1))
                
                
                #empty_code = K.tile(K.constant ( [[0,0,0,0,0]], dtype=K.floatx() ), (K.shape(self.dst_code)[0] , 1))
                empty_code = K.tile(K.constant ( [[0,0,0]], dtype=K.floatx() ), (K.shape(self.dst_code)[0] , 1))
                
                #import code
                #code.interact(local=dict(globals(), **locals()))


                #src_label = K.repeat_elements (  K.constant ( [[1,0]], dtype=K.floatx() ) , self.batch_size, axis = 0)
                #dst_label = K.repeat_elements (  K.constant ( [[0,1]], dtype=K.floatx() ) , self.batch_size, axis = 0)

                
                #self.pred_src_src = self.decoder( K.concatenate ( [ self.src_code, self.src_id, self.src_pyr]) )#, K.concatenate ( [self.src_bg_code, src_label])] )
                #self.pred_dst_dst = self.decoder( K.concatenate ( [ self.dst_code, self.dst_id, self.dst_pyr]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                #self.pred_src_dst = self.decoder( K.concatenate ( [ self.dst_code, self.src_id, self.dst_pyr]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                #self.pred_dst_src_pyr = self.decoder( K.concatenate ( [ self.dst_code, self.dst_id, self.src_pyr]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                #self.pred_est_dst_src_pyr = self.pyrest (self.encoder (self.pred_dst_src_pyr))
                
                self.pred_src_src = self.decoder( K.concatenate ( [ self.src_code, self.src_id, empty_code]) )#, K.concatenate ( [self.src_bg_code, src_label])] )
                self.pred_dst_dst = self.decoder( K.concatenate ( [ self.dst_code, self.dst_id, empty_code]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                self.pred_src_dst = self.decoder( K.concatenate ( [ self.dst_code, self.src_id, empty_code]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                self.pred_dst_src_pyr = self.decoder( K.concatenate ( [ self.dst_code, self.dst_id, empty_code]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                self.pred_est_dst_src_pyr = self.pyrest (self.encoder (self.pred_dst_src_pyr))

                self.pred_idest_src_dst = self.idest (self.encoder (self.pred_src_dst))
                
                #self.pred_src_src = self.decoder( K.concatenate ( [ self.src_code, empty_code]) )#, K.concatenate ( [self.src_bg_code, src_label])] )
                #self.pred_dst_dst = self.decoder( K.concatenate ( [ self.dst_code, empty_code]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                #self.pred_src_dst = self.decoder( K.concatenate ( [ self.dst_code, empty_code]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                #self.pred_dst_src_pyr = self.decoder( K.concatenate ( [ self.dst_code, empty_code]) )#, K.concatenate ( [self.dst_bg_code, dst_label])] )
                #self.pred_est_dst_src_pyr = self.pyrest (self.encoder (self.pred_dst_src_pyr))

                if learn_mask:
                    self.pred_src_srcm = self.decoderm(self.src_code)
                    self.pred_dst_dstm = self.decoderm(self.dst_code)
                    self.pred_src_dstm = self.decoderm(self.dst_code)

            def get_model_filename_list(self, exclude_for_pretrain=False):
                ar = []
                if not exclude_for_pretrain:
                    ar += [ [self.encoder, 'encoder.h5'] ]
                ar += [  [self.decoder, 'decoder.h5'],
                         [self.pyrest, 'pyrest.h5'],
                         [self.idest, 'idest.h5']  ]
                

                if self.learn_mask:
                    ar += [ [self.decoder_srcm, 'decoder_srcm.h5'],
                            [self.decoder_dstm, 'decoder_dstm.h5']  ]
                return ar

        self.model = SAEDFModel (self.batch_size, resolution, ae_dims, e_ch_dims, d_ch_dims, learn_mask)

        self.opt_dis_model = []

        if self.true_face_training:
            def dis_flow(ndf=256):
                def func(x):
                    x, = x

                    code_res = K.int_shape(x)[1]

                    x = Conv2D( ndf, 4, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
                    x = LeakyReLU(0.1)(x)

                    x = Conv2D( ndf*2, 3, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
                    x = LeakyReLU(0.1)(x)

                    if code_res > 8:
                        x = Conv2D( ndf*4, 3, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
                        x = LeakyReLU(0.1)(x)

                    if code_res > 16:
                        x = Conv2D( ndf*8, 3, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
                        x = LeakyReLU(0.1)(x)

                    if code_res > 32:
                        x = Conv2D( ndf*8, 3, strides=2, padding='valid')( ZeroPadding2D(1)(x) )
                        x = LeakyReLU(0.1)(x)

                    return Conv2D( 1, 1, strides=1, padding='valid', activation='sigmoid')(x)
                return func

            def dis_flow(ndf=256):
                def func(x):
                    x, = x

                    code_res = K.int_shape(x)[1]

                    x = Dense(code_res*8)(x)
                    x = LeakyReLU(0.2)(x)
                    x = Dense(code_res*4)(x)
                    x = LeakyReLU(0.2)(x)

                    return Dense( 1, activation='sigmoid')(x)
                return func

            sh = [ Input( K.int_shape(self.model.src_code)[1:] ) ]
            self.dis = modelify(dis_flow()) (sh)
            


            self.opt_dis_model = [ (self.dis, 'dis.h5') ]

        loaded, not_loaded = [], self.model.get_model_filename_list()+self.opt_dis_model
        if not self.is_first_run():
            loaded, not_loaded = self.load_weights_safe(not_loaded)

        CA_models = []
        if self.options.get('ca_weights', False):
            CA_models += [ model for model, _ in not_loaded ]

        CA_conv_weights_list = []
        for model in CA_models:
            for layer in model.layers:
                if type(layer) == keras.layers.Conv2D:
                    CA_conv_weights_list += [layer.weights[0]] #- is Conv2D kernel_weights

        if len(CA_conv_weights_list) != 0:
            CAInitializerMP ( CA_conv_weights_list )

        def CVAELoss():
            def func(input):
                mean_t, logvar_t = input
                #import code
                #code.interact(local=dict(globals(), **locals()))
                return        K.mean ( K.sum( 0.5*(K.exp(logvar_t) + K.square(mean_t)-logvar_t-1), axis=1) )
                return beta * K.mean ( K.sum( 0.5*(K.exp(logvar_t) + K.square(mean_t)-logvar_t-1), axis=1) )
            return func



        target_srcm = gaussian_blur( max(1, resolution // 32) )(self.model.target_srcm)
        target_dstm = gaussian_blur( max(1, resolution // 32) )(self.model.target_dstm)

        target_src_masked = self.model.target_src*target_srcm
        target_dst_masked = self.model.target_dst*target_dstm
        target_dst_anti_masked = self.model.target_dst*(1.0 - target_dstm)

        target_src_masked_opt = target_src_masked if masked_training else self.model.target_src
        target_dst_masked_opt = target_dst_masked if masked_training else self.model.target_dst

        pred_src_src_masked_opt = self.model.pred_src_src*target_srcm if masked_training else self.model.pred_src_src
        pred_dst_dst_masked_opt = self.model.pred_dst_dst*target_dstm if masked_training else self.model.pred_dst_dst

        psd_target_dst_masked = self.model.pred_src_dst*target_dstm
        psd_target_dst_anti_masked = self.model.pred_src_dst*(1.0 - target_dstm)

        if self.is_training_mode:
            self.src_dst_opt      = RMSprop(lr=5e-5, clipnorm=1.0 if self.options['clipgrad'] else 0.0, tf_cpu_mode=self.options['optimizer_mode']-1)
            self.src_dst_mask_opt = RMSprop(lr=5e-5, clipnorm=1.0 if self.options['clipgrad'] else 0.0, tf_cpu_mode=self.options['optimizer_mode']-1)
            self.D_opt            = RMSprop(lr=5e-5, clipnorm=1.0 if self.options['clipgrad'] else 0.0, tf_cpu_mode=self.options['optimizer_mode']-1)

            #if not self.options['pixel_loss']:
            #    src_loss = K.mean ( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( target_src_masked_opt, pred_src_src_masked_opt) )
            #else:
            #    src_loss = K.mean ( 50*K.square( target_src_masked_opt - pred_src_src_masked_opt ) )
            #
            src_loss = K.mean ( 10*K.square( target_src_masked_opt - pred_src_src_masked_opt ) ) +\
                       K.mean ( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( target_src_masked_opt, pred_src_src_masked_opt) )


            #src_loss += K.mean ( 10*K.square( self.model.src_est_pyr - self.model.src_pyr ) )

            
            face_style_power = self.options['face_style_power'] / 100.0
            if face_style_power != 0:
                src_loss += style_loss(gaussian_blur_radius=resolution//16, loss_weight=face_style_power, wnd_size=0)( psd_target_dst_masked, target_dst_masked )

            bg_style_power = self.options['bg_style_power'] / 100.0
            if bg_style_power != 0:
                if not self.options['pixel_loss']:
                    src_loss += K.mean( (10*bg_style_power)*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( psd_target_dst_anti_masked, target_dst_anti_masked ))
                else:
                    src_loss += K.mean( (50*bg_style_power)*K.square( psd_target_dst_anti_masked - target_dst_anti_masked ))

            #if not self.options['pixel_loss']:
            #    dst_loss = K.mean( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)(target_dst_masked_opt, pred_dst_dst_masked_opt) )
            #else:
            #    dst_loss = K.mean( 50*K.square( target_dst_masked_opt - pred_dst_dst_masked_opt ) )
            #
            dst_loss = K.mean ( 10*K.square( target_dst_masked_opt - pred_dst_dst_masked_opt ) ) + \
                       K.mean ( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( target_dst_masked_opt, pred_dst_dst_masked_opt) )
                       
            #dst_loss += K.mean ( 10*K.square( self.model.dst_est_pyr - self.model.dst_pyr ) )

            
            
            opt_D_loss = []
            if self.true_face_training:
                def DLoss(labels,logits):
                    return K.mean(K.binary_crossentropy(labels,logits))
                    
                true_code_d = self.dis(  K.random_normal ( K.shape(self.model.src_code) ) )
                true_code_d_ones = K.ones_like(true_code_d)

                src_code_d = self.dis( self.model.src_code )
                src_code_d_ones = K.ones_like(src_code_d)
                src_code_d_zeros = K.zeros_like(src_code_d)

                dst_code_d = self.dis( self.model.dst_code )
                dst_code_d_ones = K.ones_like(dst_code_d)
                dst_code_d_zeros = K.zeros_like(dst_code_d)

                opt_D_loss = [ 0.5*DLoss(src_code_d_ones, src_code_d)+ \
                               0.5*DLoss(dst_code_d_ones, dst_code_d) ]

                loss_D = DLoss(true_code_d_ones, true_code_d) + \
                         (DLoss(src_code_d_zeros, src_code_d ) + \
                          DLoss(dst_code_d_zeros, dst_code_d))*0.5


                self.D_train = K.function ([self.model.warped_src, self.model.warped_dst],[loss_D], self.D_opt.get_updates(loss_D, self.dis.trainable_weights) )
            
            self.src_dst_train = K.function ([self.model.warped_src, 
                                              self.model.warped_dst, self.model.target_src, self.model.target_srcm, self.model.target_dst, self.model.target_dstm,
                                              self.model.src_pyr, self.model.dst_pyr,
                                              ],
                                              [src_loss,dst_loss],
                                              self.src_dst_opt.get_updates( [src_loss+dst_loss]+opt_D_loss, self.model.enc_dec_trainable_weights)
                                             )
                                             
            src_dst_pyr_loss = K.mean ( 10*K.square( self.model.pred_est_dst_src_pyr - self.model.src_pyr ) )

            self.src_dst_pyr_train = K.function ([self.model.warped_dst, self.model.src_pyr],
                                              [src_dst_pyr_loss],
                                              self.src_dst_opt.get_updates( [src_dst_pyr_loss], self.model.enc_dec_trainable_weights)
                                             )
                                             
            id_loss = K.mean(K.binary_crossentropy(self.model.src_id, self.model.src_est_id)) + \
                      K.mean(K.binary_crossentropy(self.model.dst_id, self.model.dst_est_id))
                      
            self.src_dst_id_train = K.function ([self.model.warped_src, self.model.warped_dst],
                                                [id_loss],
                                                self.src_dst_opt.get_updates( [id_loss], self.model.idest_trainable_weights)
                                               )     
                                             
            id_loss = K.mean( K.binary_crossentropy(self.model.src_id, self.model.pred_idest_src_dst))                                                           

            self.src_dst_force_id_train = K.function ([self.model.warped_dst],
                                              [id_loss],
                                              self.src_dst_opt.get_updates( [id_loss], self.model.decoder_trainable_weights)
                                             )              
            
            if self.options['learn_mask']:
                src_mask_loss = K.mean(K.square(self.model.target_srcm-self.model.pred_src_srcm))
                dst_mask_loss = K.mean(K.square(self.model.target_dstm-self.model.pred_dst_dstm))
                self.src_dst_mask_train = K.function ([self.model.warped_src, self.model.warped_dst, self.model.target_srcm, self.model.target_dstm],[src_mask_loss, dst_mask_loss], self.src_dst_mask_opt.get_updates(src_mask_loss+dst_mask_loss, self.model.src_dst_mask_trainable_weights ) )

            if self.options['learn_mask']:
                self.AE_view = K.function ([self.model.warped_src, self.model.warped_dst], [self.model.pred_src_src, self.model.pred_dst_dst, self.model.pred_dst_dstm, self.model.pred_src_dst, self.model.pred_src_dstm])
            else:
                self.AE_view = K.function ([self.model.warped_src, self.model.src_pyr, self.model.warped_dst, self.model.dst_pyr], 
                                           [self.model.pred_src_src, self.model.pred_dst_dst, self.model.pred_src_dst, self.model.pred_dst_src_pyr ])

        else:
            if self.options['learn_mask']:
                self.AE_convert = K.function ([self.model.warped_dst],[ self.model.pred_src_dst, self.model.pred_dst_dstm, self.model.pred_src_dstm ])
            else:
                self.AE_convert = K.function ([self.model.warped_dst],[ self.model.pred_src_dst ])


        if self.is_training_mode:
            t = SampleProcessor.Types

            if self.options['face_type'] == 'h':
                face_type = t.FACE_TYPE_HALF
            elif self.options['face_type'] == 'mf':
                face_type = t.FACE_TYPE_MID_FULL
            elif self.options['face_type'] == 'f':
                face_type = t.FACE_TYPE_FULL

            t_mode_bgr = t.MODE_BGR if not self.pretrain else t.MODE_BGR_SHUFFLE

            training_data_src_path = self.training_data_src_path
            training_data_dst_path = self.training_data_dst_path
            sort_by_yaw = self.sort_by_yaw

            if self.pretrain and self.pretraining_data_path is not None:
                training_data_src_path = self.pretraining_data_path
                training_data_dst_path = self.pretraining_data_path
                sort_by_yaw = False

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, sort_by_yaw_target_samples_path=training_data_dst_path if sort_by_yaw else None,
                                                                random_ct_samples_path=training_data_dst_path if apply_random_ct else None,
                                                                debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0], scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
                        output_sample_types = [ {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution':resolution, 'apply_ct': apply_random_ct},
                                                #{'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': resolution, 'apply_ct': apply_random_ct },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution },
                                                {'types': (t.IMG_PITCH_YAW_ROLL,)}  ]
                         ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=False, rotation_range=[0,0], ),
                        output_sample_types = [ {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution':resolution},
                                                #{'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution},
                                                {'types': (t.IMG_PITCH_YAW_ROLL,)}  ])
                             ])

    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list ( exclude_for_pretrain=(self.pretrain and self.iter != 0) ) +self.opt_dis_model

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list()+self.opt_dis_model )

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, target_src, target_srcm, src_pyr = generators_samples[0]
        warped_dst, target_dst, target_dstm, dst_pyr = generators_samples[1]
              
        feed = [warped_src, warped_dst, target_src, target_srcm, target_dst, target_dstm ]

        a = 60 / 2
        src_pyr = np.round( (src_pyr+1) * a )  / a -1 #.astype(K.floatx())
        feed += [src_pyr] 
        
        dst_pyr = np.round( (dst_pyr+1) * a )  / a -1 #.astype(K.floatx())
        feed += [dst_pyr] 
        
        #import code
        #code.interact(local=dict(globals(), **locals()))
        
        src_loss, dst_loss,  = self.src_dst_train (feed)
        
        #dst_pyr_loss, = self.src_dst_pyr_train( [warped_dst, src_pyr] )
        #dst_loss += dst_pyr_loss
        
        id_loss = self.src_dst_id_train( [warped_src, warped_dst] )[0]
        id_loss += self.src_dst_force_id_train([warped_dst])[0]
            
        if self.true_face_training:
            self.D_train([warped_src, warped_dst, ])

        if self.options['learn_mask']:
            feed = [ warped_src, warped_dst, target_srcm, target_dstm ]
            src_mask_loss, dst_mask_loss, = self.src_dst_mask_train (feed)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), ('id_loss', id_loss) )

    #override
    def onGetPreview(self, sample):
        test_S     = sample[0][1][0:4] #first 4 samples
        test_S_m   = sample[0][2][0:4] #first 4 samples
        test_S_pyr = sample[0][3][0:4]
        
        test_D     = sample[1][1][0:4]
        test_D_m   = sample[1][2][0:4]
        test_D_pyr = sample[1][3][0:4]
        
        a = 60 / 2
        test_S_pyr = np.round( (test_S_pyr+1) * a )  / a -1 
        test_D_pyr = np.round( (test_D_pyr+1) * a )  / a -1
        
        if self.options['learn_mask']:
            S, D, SS, DD, DDM, SD, SDM = [ np.clip(x, 0.0, 1.0) for x in ([test_S,test_D] + self.AE_view ([test_S, test_S_m, test_D, test_D_m]) ) ]
            DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]
        else:
            S, D, SS, DD, SD, DSPyr = [ np.clip(x, 0.0, 1.0) for x in ([test_S,test_D] + self.AE_view ([test_S, test_S_pyr, test_D, test_D_pyr]) ) ]

        result = []
        st = []
        for i in range(len(test_S)):
            ar = S[i], SS[i], D[i], DD[i], SD[i], DSPyr[i]

            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('SAE', np.concatenate (st, axis=0 )), ]

        if self.options['learn_mask']:
            st_m = []
            for i in range(len(test_S)):
                ar = S[i]*test_S_m[i], SS[i], D[i]*test_D_m[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('SAE masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.AE_convert ([ np.zeros ( (1, self.options['resolution'], self.options['resolution'], 3), dtype=np.float32 ) ])
        else:
            if self.options['learn_mask']:
                bgr, mask_dst_dstm, mask_src_dstm = self.AE_convert ([face[np.newaxis,...]])
                mask = mask_dst_dstm[0] * mask_src_dstm[0]
                return bgr[0], mask[...,0]
            else:
                bgr, = self.AE_convert ([face[np.newaxis,...]])
                return bgr[0]

    #override
    def get_ConverterConfig(self):
        if self.options['face_type'] == 'h':
            face_type = FaceType.HALF
        elif self.options['face_type'] == 'mf':
            face_type = FaceType.MID_FULL
        elif self.options['face_type'] == 'f':
            face_type = FaceType.FULL

        import converters
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), converters.ConverterConfigMasked(face_type=face_type,
                                     default_mode = 1 if self.options['apply_random_ct'] or self.options['face_style_power'] or self.options['bg_style_power'] else 4,
                                     clip_hborder_mask_per=0.0625 if (face_type == FaceType.FULL) else 0,
                                    )

Model = SAEv2Model
