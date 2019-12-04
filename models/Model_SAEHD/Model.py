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
            resolution = io.input_int("分辨率 ( 64-1024 帮助:? 跳过:128) : ", default_resolution, help_message="更高的分辨率需要更多的VRAM和训练时间。 数值调整成16的倍数。")
            resolution = np.clip (resolution, 64, 1024)
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

                        return Conv2D(output_nc, kernel_size=5, padding='same', activation='sigmoid')(x)

                    return func

                self.encoder = modelify(enc_flow(e_ch_dims, ae_dims, lowest_dense_res)) ( Input(bgr_shape) )

                sh = K.int_shape( self.encoder.outputs[0] )[1:]
                self.decoder_src = modelify(dec_flow(output_nc, d_ch_dims)) ( Input(sh) )
                self.decoder_dst = modelify(dec_flow(output_nc, d_ch_dims)) ( Input(sh) )

                if learn_mask:
                    self.decoder_srcm = modelify(dec_flow(1, d_ch_dims, is_mask=True)) ( Input(sh) )
                    self.decoder_dstm = modelify(dec_flow(1, d_ch_dims, is_mask=True)) ( Input(sh) )

                self.src_dst_trainable_weights = self.encoder.trainable_weights + self.decoder_src.trainable_weights + self.decoder_dst.trainable_weights

                if learn_mask:
                    self.src_dst_mask_trainable_weights = self.encoder.trainable_weights + self.decoder_srcm.trainable_weights + self.decoder_dstm.trainable_weights

                self.warped_src, self.warped_dst = Input(bgr_shape), Input(bgr_shape)
                self.target_src, self.target_dst = Input(bgr_shape), Input(bgr_shape)
                self.target_srcm, self.target_dstm = Input(mask_shape), Input(mask_shape)
                self.src_code, self.dst_code = self.encoder(self.warped_src), self.encoder(self.warped_dst)

                self.pred_src_src = self.decoder_src(self.src_code)
                self.pred_dst_dst = self.decoder_dst(self.dst_code)
                self.pred_src_dst = self.decoder_src(self.dst_code)

                if learn_mask:
                    self.pred_src_srcm = self.decoder_srcm(self.src_code)
                    self.pred_dst_dstm = self.decoder_dstm(self.dst_code)
                    self.pred_src_dstm = self.decoder_srcm(self.dst_code)

            def get_model_filename_list(self, exclude_for_pretrain=False):
                ar = []
                if not exclude_for_pretrain:
                    ar += [ [self.encoder, 'encoder.h5'] ]
                ar += [  [self.decoder_src, 'decoder_src.h5'],
                         [self.decoder_dst, 'decoder_dst.h5']  ]
                if self.learn_mask:
                    ar += [ [self.decoder_srcm, 'decoder_srcm.h5'],
                            [self.decoder_dstm, 'decoder_dstm.h5']  ]
                return ar

        class SAELIAEModel(CommonModel):
            def __init__(self, resolution, ae_dims, e_ch_dims, d_ch_dims, learn_mask):
                super().__init__()
                self.learn_mask = learn_mask

                output_nc = 3
                bgr_shape = (resolution, resolution, output_nc)
                mask_shape = (resolution, resolution, 1)

                lowest_dense_res = resolution // 16

                def enc_flow(e_ch_dims):
                    dims = output_nc*e_ch_dims
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

                        x = Flatten()(x)
                        return x
                    return func

                def inter_flow(lowest_dense_res, ae_dims):
                    def func(x):
                        x = Dense(ae_dims)(x)
                        x = Dense(lowest_dense_res * lowest_dense_res * ae_dims*2)(x)
                        x = Reshape((lowest_dense_res, lowest_dense_res, ae_dims*2))(x)
                        x = self.upscale(ae_dims*2)(x)
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

                        return Conv2D(output_nc, kernel_size=5, padding='same', activation='sigmoid')(x)

                    return func

                self.encoder = modelify(enc_flow(e_ch_dims)) ( Input(bgr_shape) )

                sh = K.int_shape( self.encoder.outputs[0] )[1:]
                self.inter_B = modelify(inter_flow(lowest_dense_res, ae_dims)) ( Input(sh) )
                self.inter_AB = modelify(inter_flow(lowest_dense_res, ae_dims)) ( Input(sh) )

                sh = np.array(K.int_shape( self.inter_B.outputs[0] )[1:])*(1,1,2)
                self.decoder = modelify(dec_flow(output_nc, d_ch_dims)) ( Input(sh) )

                if learn_mask:
                    self.decoderm = modelify(dec_flow(1, d_ch_dims, is_mask=True)) ( Input(sh) )

                self.src_dst_trainable_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoder.trainable_weights

                if learn_mask:
                    self.src_dst_mask_trainable_weights = self.encoder.trainable_weights + self.inter_B.trainable_weights + self.inter_AB.trainable_weights + self.decoderm.trainable_weights

                self.warped_src, self.warped_dst = Input(bgr_shape), Input(bgr_shape)
                self.target_src, self.target_dst = Input(bgr_shape), Input(bgr_shape)
                self.target_srcm, self.target_dstm = Input(mask_shape), Input(mask_shape)

                warped_src_code = self.encoder (self.warped_src)
                warped_src_inter_AB_code = self.inter_AB (warped_src_code)
                self.src_code = Concatenate()([warped_src_inter_AB_code,warped_src_inter_AB_code])

                warped_dst_code = self.encoder (self.warped_dst)
                warped_dst_inter_B_code = self.inter_B (warped_dst_code)
                warped_dst_inter_AB_code = self.inter_AB (warped_dst_code)
                self.dst_code = Concatenate()([warped_dst_inter_B_code,warped_dst_inter_AB_code])

                src_dst_code = Concatenate()([warped_dst_inter_AB_code,warped_dst_inter_AB_code])

                self.pred_src_src = self.decoder(self.src_code)
                self.pred_dst_dst = self.decoder(self.dst_code)
                self.pred_src_dst = self.decoder(src_dst_code)

                if learn_mask:
                    self.pred_src_srcm = self.decoderm(self.src_code)
                    self.pred_dst_dstm = self.decoderm(self.dst_code)
                    self.pred_src_dstm = self.decoderm(src_dst_code)

            def get_model_filename_list(self, exclude_for_pretrain=False):
                ar = [ [self.encoder, 'encoder.h5'],
                       [self.inter_B, 'inter_B.h5'] ]

                if not exclude_for_pretrain:
                    ar += [ [self.inter_AB, 'inter_AB.h5'] ]

                ar += [  [self.decoder, 'decoder.h5']  ]

                if self.learn_mask:
                    ar += [ [self.decoderm, 'decoderm.h5'] ]

                return ar

        if 'df' in self.options['archi']:
            self.model = SAEDFModel (resolution, ae_dims, ed_ch_dims, ed_ch_dims, learn_mask)
        elif 'liae' in self.options['archi']:
            self.model = SAELIAEModel (resolution, ae_dims, ed_ch_dims, ed_ch_dims, learn_mask)

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

            sh = [ Input( K.int_shape(self.model.src_code)[1:] ) ]
            self.dis = modelify(dis_flow()) (sh)

            self.opt_dis_model = [ (self.dis, 'dis.h5') ]

        loaded, not_loaded = [], self.model.get_model_filename_list()+self.opt_dis_model
        if not self.is_first_run():
            loaded, not_loaded = self.load_weights_safe(not_loaded)

        CA_models = [ model for model, _ in not_loaded ]

        self.CA_conv_weights_list = []
        for model in CA_models:
            for layer in model.layers:
                if type(layer) == keras.layers.Conv2D:
                    self.CA_conv_weights_list += [layer.weights[0]] #- is Conv2D kernel_weights

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
        
            src_loss =  K.mean ( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( target_src_masked_opt, pred_src_src_masked_opt) )
            src_loss += K.mean ( 10*K.square( target_src_masked_opt - pred_src_src_masked_opt ) )

            face_style_power = self.options['face_style_power'] / 100.0
            if face_style_power != 0:
                src_loss += style_loss(gaussian_blur_radius=resolution//16, loss_weight=face_style_power, wnd_size=0)( psd_target_dst_masked, target_dst_masked )

            bg_style_power = self.options['bg_style_power'] / 100.0
            if bg_style_power != 0:
                src_loss += K.mean( (10*bg_style_power)*dssim(kernel_size=int(resolution/11.6),max_value=1.0)( psd_target_dst_anti_masked, target_dst_anti_masked ))
                src_loss += K.mean( (10*bg_style_power)*K.square( psd_target_dst_anti_masked - target_dst_anti_masked ))

            dst_loss =  K.mean( 10*dssim(kernel_size=int(resolution/11.6),max_value=1.0)(target_dst_masked_opt, pred_dst_dst_masked_opt) )
            dst_loss += K.mean( 10*K.square( target_dst_masked_opt - pred_dst_dst_masked_opt ) )

            G_loss = src_loss+dst_loss

            if self.true_face_training:
                def DLoss(labels,logits):
                    return K.mean(K.binary_crossentropy(labels,logits))

                src_code_d = self.dis( self.model.src_code )
                src_code_d_ones = K.ones_like(src_code_d)
                src_code_d_zeros = K.zeros_like(src_code_d)
                dst_code_d = self.dis( self.model.dst_code )
                dst_code_d_ones = K.ones_like(dst_code_d)
                G_loss += 0.01*DLoss(src_code_d_ones, src_code_d)

                loss_D = (DLoss(dst_code_d_ones , dst_code_d) + \
                          DLoss(src_code_d_zeros, src_code_d) ) * 0.5

                self.D_train = K.function ([self.model.warped_src, self.model.warped_dst],[loss_D], self.D_opt.get_updates(loss_D, self.dis.trainable_weights) )

            self.src_dst_train = K.function ([self.model.warped_src, self.model.warped_dst, self.model.target_src, self.model.target_srcm, self.model.target_dst, self.model.target_dstm],
                                             [src_loss,dst_loss],
                                             self.src_dst_opt.get_updates( G_loss, self.model.src_dst_trainable_weights)
                                             )

            if self.options['learn_mask']:
                src_mask_loss = K.mean(K.square(self.model.target_srcm-self.model.pred_src_srcm))
                dst_mask_loss = K.mean(K.square(self.model.target_dstm-self.model.pred_dst_dstm))
                self.src_dst_mask_train = K.function ([self.model.warped_src, self.model.warped_dst, self.model.target_srcm, self.model.target_dstm],[src_mask_loss, dst_mask_loss], self.src_dst_mask_opt.get_updates(src_mask_loss+dst_mask_loss, self.model.src_dst_mask_trainable_weights ) )

            if self.options['learn_mask']:
                self.AE_view = K.function ([self.model.warped_src, self.model.warped_dst], [self.model.pred_src_src, self.model.pred_dst_dst, self.model.pred_dst_dstm, self.model.pred_src_dst, self.model.pred_src_dstm])
            else:
                self.AE_view = K.function ([self.model.warped_src, self.model.warped_dst], [self.model.pred_src_src, self.model.pred_dst_dst, self.model.pred_src_dst ])

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
                        output_sample_types = [ {'types' : (t_img_warped, face_type, t_mode_bgr), 'resolution':resolution, 'ct_mode': self.options['ct_mode'] },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': resolution, 'ct_mode': self.options['ct_mode'] },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution } ]
                                              ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                        sample_process_options=SampleProcessor.Options(random_flip=self.random_flip, ),
                        output_sample_types = [ {'types' : (t_img_warped, face_type, t_mode_bgr), 'resolution':resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t_mode_bgr), 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_M), 'resolution': resolution} ])
                             ])

    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list ( exclude_for_pretrain=(self.pretrain and self.iter != 0) ) +self.opt_dis_model

    #override
    def onSave(self):
        self.save_weights_safe( self.get_model_filename_list()+self.opt_dis_model )

    #override
    def on_success_train_one_iter(self):
        if len(self.CA_conv_weights_list) != 0:
            exec(nnlib.import_all(), locals(), globals())
            CAInitializerMP ( self.CA_conv_weights_list )
            self.CA_conv_weights_list = []

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, target_src, target_srcm = generators_samples[0]
        warped_dst, target_dst, target_dstm = generators_samples[1]

        feed = [warped_src, warped_dst, target_src, target_srcm, target_dst, target_dstm]

        src_loss, dst_loss, = self.src_dst_train (feed)

        if self.true_face_training:
            self.D_train([warped_src, warped_dst])

        if self.options['learn_mask']:
            feed = [ warped_src, warped_dst, target_srcm, target_dstm ]
            src_mask_loss, dst_mask_loss, = self.src_dst_mask_train (feed)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, sample):
        test_S   = sample[0][1][0:4] #first 4 samples
        test_S_m = sample[0][2][0:4] #first 4 samples
        test_D   = sample[1][1][0:4]
        test_D_m = sample[1][2][0:4]

        if self.options['learn_mask']:
            S, D, SS, DD, DDM, SD, SDM = [ np.clip(x, 0.0, 1.0) for x in ([test_S,test_D] + self.AE_view ([test_S, test_D]) ) ]
            DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]
        else:
            S, D, SS, DD, SD, = [ np.clip(x, 0.0, 1.0) for x in ([test_S,test_D] + self.AE_view ([test_S, test_D]) ) ]

        result = []
        st = []
        for i in range(len(test_S)):
            ar = S[i], SS[i], D[i], DD[i], SD[i]

            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('SAEHD', np.concatenate (st, axis=0 )), ]

        if self.options['learn_mask']:
            st_m = []
            for i in range(len(test_S)):
                ar = S[i]*test_S_m[i], SS[i], D[i]*test_D_m[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('SAEHD masked', np.concatenate (st_m, axis=0 )), ]

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
