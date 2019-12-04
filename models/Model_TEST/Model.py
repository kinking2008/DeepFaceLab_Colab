from functools import partial

import numpy as np

import mathlib
from facelib import FaceType
from interact import interact as io
from models import ModelBase
from nnlib import nnlib
from samplelib import *


#SAE - Styled AutoEncoder
class SAEHDModel(ModelBase):

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        yn_str = {True:'y',False:'n'}

        default_resolution = 128
        default_archi = 'df'
        default_face_type = 'f'
        

        if is_first_run:
            resolution = io.input_int("分辨率 ( 64-256 帮助:? 跳过:128) : ", default_resolution, help_message="更高的分辨率需要更多的VRAM和训练时间。 数值调整成16的倍数。")
            resolution = np.clip (resolution, 64, 256)
            while np.modf(resolution / 16)[0] != 0.0:
                resolution -= 1
            self.options['resolution'] = resolution
            self.options['face_type'] = io.input_str ("半脸（h）,中脸（mf）,全脸（f）? (帮助:? 跳过:f): ", default_face_type, ['h','mf','f'], help_message="半脸具有更好的分辨率，但脸颊面积较小。中脸比半脸宽30％。").lower()
        else:
            self.options['resolution'] = self.options.get('resolution', default_resolution)
            self.options['face_type'] = self.options.get('face_type', default_face_type)

        default_learn_mask = self.options.get('learn_mask', True)
        if is_first_run or ask_override:
            self.options['learn_mask'] = io.input_bool ( f"学习面具? (y/n, 帮助:? 跳过:{yn_str[default_learn_mask]} ) : " , default_learn_mask, help_message="学习遮罩可以帮助模型识别人脸方向。没有遮罩的学习可以减小模型大小，在这种情况下，转换器被迫使用不像预期的那样平滑的'未预测的遮罩'。")
        else:
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

        default_ae_dims = 256
        default_ed_ch_dims = 21

        if is_first_run:
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder变暗 (32-1024 帮助:? 跳过:%d) : " % (default_ae_dims) , default_ae_dims, help_message="所有面部信息都将打包到AE暗淡。如果AE暗淡的数量不够，那么例如闭合的眼睛将无法识别。更多的暗淡更好，但需要更多的VRAM。您可以微调模型大小以适合您的GPU。" ), 32, 1024 )
            self.options['ed_ch_dims'] = np.clip ( io.input_int("每个通道的编码器/解码器暗淡 (10-85 帮助:? 跳过:%d) : " % (default_ed_ch_dims) , default_ed_ch_dims, help_message="更多的暗淡有助于识别更多的面部特征并获得更清晰的效果，但需要更多的VRAM。您可以微调模型大小以适合您的GPU" ), 10, 85 )
        else:
            self.options['ae_dims'] = self.options.get('ae_dims', default_ae_dims)
            self.options['ed_ch_dims'] = self.options.get('ed_ch_dims', default_ed_ch_dims)

        default_true_face_training = self.options.get('true_face_training', False)
        default_face_style_power = self.options.get('face_style_power', 0.0)
        default_bg_style_power = self.options.get('bg_style_power', 0.0)
        
        if is_first_run or ask_override:
            default_random_warp = self.options.get('random_warp', True)
            self.options['random_warp'] = io.input_bool (f"启用样本的随机扭曲吗? ( y/n, 帮助:? 跳过:{yn_str[default_random_warp]}) : ", default_random_warp, help_message="需要随机扭曲来概括两张脸的面部表情。当对脸部进行足够的训练后，可以禁用它以在减少迭代次数的情况下获得额外的清晰度。")

            self.options['true_face_training'] = io.input_bool (f"启用 '真脸' 训练? (y/n, 帮助:? 跳过:{yn_str[default_true_face_training]}) : ", default_true_face_training, help_message="使用结果脸将更像src并将获得额外的清晰度。转换前开启运行10-20k次迭代后再转换。")

            self.options['face_style_power'] = np.clip ( io.input_number("面部风格样式 ( 0.0 .. 100.0 帮助:? 跳过:%.2f) : " % (default_face_style_power), default_face_style_power,
                                                                               help_message="学会转移面部样式细节，例如光线和颜色条件。警告：只有在10k iters之后启用它，当预测的面部足够清晰以开始学习风格时。从0.1值开始并检查历史记录更改。启用此选项会增加 模特崩溃。"), 0.0, 100.0 )

            self.options['bg_style_power'] = np.clip ( io.input_number("背景风格样式( 0.0 .. 100.0 帮助:? 跳过:%.2f) : " % (default_bg_style_power), default_bg_style_power,
                                                                               help_message="学会围绕脸部传递图像。这可以让脸部更像dst。启用此选项会增加模型崩溃的机会。"), 0.0, 100.0 )

            default_ct_mode = self.options.get('ct_mode', 'none')
            self.options['ct_mode'] = io.input_str (f"选择颜色转换模式应用于src脸部设置. ( none/rct/lct/mkl/idt/sot, 帮助:? 跳过:{default_ct_mode}) : ", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="更改drc样本附近的src样本的颜色分布。尝试所有模式以找到最佳的颜色。")
            
            if nnlib.device.backend != 'plaidML': # todo https://github.com/plaidml/plaidml/issues/301
                default_clipgrad = False if is_first_run else self.options.get('clipgrad', False)
                self.options['clipgrad'] = io.input_bool (f"启用渐变剪辑? (y/n, 帮助:? 跳过:{yn_str[default_clipgrad]}) : ", default_clipgrad, help_message="渐变裁剪减少了模型崩溃的可能性，牺牲了训练速度.")
            else:
                self.options['clipgrad'] = False

        else:
            self.options['random_warp'] = self.options.get('random_warp', True)
            self.options['true_face_training'] = self.options.get('true_face_training', default_true_face_training)
            self.options['face_style_power'] = self.options.get('face_style_power', default_face_style_power)
            self.options['bg_style_power'] = self.options.get('bg_style_power', default_bg_style_power)
            self.options['ct_mode'] = self.options.get('ct_mode', 'none')
            self.options['clipgrad'] = self.options.get('clipgrad', False)

        if is_first_run:
            self.options['pretrain'] = io.input_bool ("预训练? (y/n, 帮助:? 跳过:n) : ", False, help_message="通过加载pretrain_CelebA目录下人脸进行预训练。这种技术可能有助于训练具有过度不同的面部形状和src / dst数据的光照条件的伪造生成。用这种方式脸部看起来会有些变形，为了减少这种变形效果，一些模型文件将被初始化但在预训练后不会更新：LIAE：inter_AB.h5 DF：encoder.h5。预训练模型的时间越长，变形的脸就会越多。 然后，保存并再次运行模型就会进入常规模式。")
        else:
            self.options['pretrain'] = False

    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements({1.5:4,4:8})

        resolution = self.options['resolution']
        learn_mask = self.options['learn_mask']

        ae_dims = self.options['ae_dims']
        ed_ch_dims = self.options['ed_ch_dims']
        self.pretrain = self.options['pretrain'] = self.options.get('pretrain', False)
        if not self.pretrain:
            self.options.pop('pretrain')

        bgr_shape = (resolution, resolution, 3)
        mask_shape = (resolution, resolution, 1)

        self.true_face_training = self.options.get('true_face_training', False)
        masked_training = True

        class CommonModel(object):
            def downscale (self, dim, kernel_size=5, dilation_rate=1, use_activator=True):
                def func(x):
                    if not use_activator:
                        return SubpixelDownscaler()(Conv2D(dim // 4, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x))
                    else:
                        return SubpixelDownscaler()(LeakyReLU(0.1)(Conv2D(dim // 4, kernel_size=kernel_size, strides=1, dilation_rate=dilation_rate, padding='same')(x)))
                return func

            def upscale (self, dim, size=(2,2)):
                def func(x):
                    return SubpixelUpscaler(size=size)(LeakyReLU(0.1)(Conv2D(dim * np.prod(size) , kernel_size=3, strides=1, padding='same')(x)))
                return func

            def ResidualBlock(self, dim):
                def func(inp):
                    x = Conv2D(dim, kernel_size=3, padding='same')(inp)
                    x = LeakyReLU(0.2)(x)
                    x = Conv2D(dim, kernel_size=3, padding='same')(x)
                    x = Add()([x, inp])
                    x = LeakyReLU(0.2)(x)
                    return x
                return func

        class SAEDFModel(CommonModel):
            def __init__(self, resolution, ae_dims, e_ch_dims, d_ch_dims, learn_mask):
                super().__init__()
                self.learn_mask = learn_mask

                output_nc = 3
                bgr_shape = (resolution, resolution, output_nc)
                mask_shape = (resolution, resolution, 1)
                lowest_dense_res = resolution // 16
                e_dims = output_nc*e_ch_dims



                def enc_flow(e_ch_dims, ae_dims, lowest_dense_res):
                    dims = output_nc * e_ch_dims
                    if dims % 2 != 0:
                        dims += 1

                    def func(inp):
                        x = self.downscale(dims  , 3, 1 )(inp)
                        x = self.downscale(dims*2, 3, 1 )(x)
                        x = self.downscale(dims*4, 3, 1 )(x)
                        x0 = self.downscale(dims*8, 3, 1 )(x)

                        x = self.downscale(dims  , 5, 1 )(inp)
                        x = self.downscale(dims*2, 5, 1 )(x)
                        x = self.downscale(dims*4, 5, 1 )(x)
                        x1 = self.downscale(dims*8, 5, 1 )(x)

                        x = self.downscale(dims  , 5, 2 )(inp)
                        x = self.downscale(dims*2, 5, 2 )(x)
                        x = self.downscale(dims*4, 5, 2 )(x)
                        x2 = self.downscale(dims*8, 5, 2 )(x)

                        x = self.downscale(dims  , 7, 2 )(inp)
                        x = self.downscale(dims*2, 7, 2 )(x)
                        x = self.downscale(dims*4, 7, 2 )(x)
                        x3 = self.downscale(dims*8, 7, 2 )(x)

                        x = Concatenate()([x0,x1,x2,x3])

                        x = Dense(ae_dims)(Flatten()(x))
                        x = Dense(lowest_dense_res * lowest_dense_res * ae_dims)(x)
                        x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims))(x)
                        x = self.upscale(ae_dims)(x)
                        return x
                    return func

                def dec_flow(output_nc, d_ch_dims, is_mask=False):
                    dims = output_nc * d_ch_dims
                    if dims % 2 != 0:
                        dims += 1



                    def func(x):

                        for i in [8,4,2]:
                            x = self.upscale(dims*i)(x)

                            if not is_mask:
                                x0 = x
                                x = self.upscale( (dims*i)//2 )(x)
                                x = self.ResidualBlock( (dims*i)//2 )(x)
                                x = self.downscale( dims*i, use_activator=False ) (x)
                                x = Add()([x, x0])
                                x = LeakyReLU(0.2)(x)

                        albedo = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
                        shade = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(x)
                        
                        return albedo,shade

                    return func

                self.encoder = modelify(enc_flow(e_ch_dims, ae_dims, lowest_dense_res)) ( Input(bgr_shape) )

                sh = K.int_shape( self.encoder.outputs[0] )[1:]
                self.decoder = modelify(dec_flow(output_nc, d_ch_dims)) ( Input(sh) )

                self.enc_dec_trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights

                self.target = Input(bgr_shape)
                self.code = self.encoder(self.target)

                self.pred_albedo, self.pred_shade = self.decoder(self.code)
                

            def get_model_filename_list(self, exclude_for_pretrain=False):
                ar = []
                if not exclude_for_pretrain:
                    ar += [ [self.encoder, 'encoder.h5'] ]
                ar += [  [self.decoder, 'decoder.h5'] ]
                
                return ar

        if 'df' in self.options['archi']:
            self.model = SAEDFModel (resolution, ae_dims, ed_ch_dims, ed_ch_dims, learn_mask)

        self.opt_dis_model = []

        loaded, not_loaded = [], self.model.get_model_filename_list()+self.opt_dis_model
        if not self.is_first_run():
            loaded, not_loaded = self.load_weights_safe(not_loaded)

        CA_models = [ model for model, _ in not_loaded ]

        self.CA_conv_weights_list = []
        for model in CA_models:
            for layer in model.layers:
                if type(layer) == keras.layers.Conv2D:
                    self.CA_conv_weights_list += [layer.weights[0]] #- is Conv2D kernel_weights


        if self.is_training_mode:
            self.src_dst_opt      = RMSprop(lr=5e-5, clipnorm=1.0 if self.options['clipgrad'] else 0.0, tf_cpu_mode=self.options['optimizer_mode']-1)

            alb, sh = self.model.pred_albedo, self.model.pred_shade
            
            
            
            sh_blur = gaussian_blur(1.0)(sh)
            
            G_loss =  K.mean ( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( self.model.target, alb*sh_blur ) ) + \
                      K.mean ( 10* K.square( self.model.target-alb*sh_blur ) ) + \
                      K.mean( K.var(alb,axis=[0,3]))#+ \
                      #K.mean( K.var(sh,axis=[0,3]))

            self.enc_dec_train = K.function ([self.model.target],
                                             [G_loss],
                                             self.src_dst_opt.get_updates( G_loss, self.model.enc_dec_trainable_weights)
                                             )

            self.AE_view = K.function ([self.model.target], [alb,sh,alb*sh])

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
                
            t_img_warped = t.IMG_WARPED_TRANSFORMED if self.options['random_warp'] else t.IMG_TRANSFORMED 

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, sort_by_yaw_target_samples_path=training_data_dst_path if sort_by_yaw else None,
                                                                random_ct_samples_path=training_data_dst_path if self.options['ct_mode'] != 'none' else None,
                                                                debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, scale_range=np.array([-0.05, 0.05])+self.src_scale_mod / 100.0 ),
                        output_sample_types = [ {'types' : (t.IMG_SOURCE, face_type, t_mode_bgr), 'resolution': resolution, 'ct_mode': self.options['ct_mode'] },
                                            ]),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, ),
                        output_sample_types = [ {'types' : (t.IMG_SOURCE, face_type, t_mode_bgr), 'resolution': resolution} ]
                                                )
                             ])

    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list ( exclude_for_pretrain=(self.pretrain and self.iter != 0) ) +self.opt_dis_model

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list()+self.opt_dis_model )

    #override
    def on_success_train_one_iter(self):
        return
        if len(self.CA_conv_weights_list) != 0:
            exec(nnlib.import_all(), locals(), globals())
            CAInitializerMP ( self.CA_conv_weights_list )
            self.CA_conv_weights_list = []

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        target_src,  = generators_samples[0]
        target_dst,  = generators_samples[1]


        src_loss = self.enc_dec_train ([target_src, target_src])[0]
        src_loss += self.enc_dec_train ([target_dst, target_dst])[0]


        return ( ('src_loss', src_loss), )

    #override
    def onGetPreview(self, sample):
        test_S   = sample[0][0][0:4] #first 4 samples
        test_D   = sample[1][0][0:4]

        S, S_alb, S_sh, SS = [ np.clip(x, 0.0, 1.0) for x in ([test_S] + self.AE_view ([test_S]) ) ]
        D, D_alb, D_sh, DD = [ np.clip(x, 0.0, 1.0) for x in ([test_D] + self.AE_view ([test_D]) ) ]
        
        S_sh = np.repeat (S_sh, 3, -1)
        D_sh = np.repeat (D_sh, 3, -1)
        result = []
        st = []
        
        for i in range(len(test_S)):
            ar = S[i], S_alb[i], S_sh[i], SS[i], D[i], D_alb[i], D_sh[i], DD[i]

            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('TEST', np.concatenate (st, axis=0 )), ]

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
                                     default_mode = 'overlay' if self.options['ct_mode'] != 'none' or self.options['face_style_power'] or self.options['bg_style_power'] else 'seamless',
                                     clip_hborder_mask_per=0.0625 if (face_type != FaceType.HALF) else 0,
                                    )

Model = SAEHDModel
