import numpy as np
import copy

from facelib import FaceType
from interact import interact as io


class ConverterConfig(object):
    TYPE_NONE = 0
    TYPE_MASKED = 1
    TYPE_FACE_AVATAR = 2
    ####

    TYPE_IMAGE = 3
    TYPE_IMAGE_WITH_LANDMARKS = 4

    def __init__(self, type=0,

                       super_resolution_mode=0,
                       sharpen_mode=0,
                       blursharpen_amount=0,
                       **kwargs
                       ):
        self.type = type

        self.superres_func = None
        self.blursharpen_func = None
        self.fanseg_input_size = None
        self.fanseg_extract_func = None

        self.fanchq_input_size = None
        self.fanchq_extract_func = None
        self.ebs_ct_func = None

        self.super_res_dict = {0:"None", 1:'RankSRGAN'}
        self.sharpen_dict = {0:"None", 1:'box', 2:'gaussian'}

        #default changeable params
        self.super_resolution_mode = super_resolution_mode
        self.sharpen_mode = sharpen_mode
        self.blursharpen_amount = blursharpen_amount

    def copy(self):
        return copy.copy(self)

    #overridable
    def ask_settings(self):
        s = """选择锐化模式: \n"""
        for key in self.sharpen_dict.keys():
            s += f"""({key}) {self.sharpen_dict[key]}\n"""
        s += f"""帮助:? 默认: {list(self.sharpen_dict.keys())[0]} : """
        self.sharpen_mode = io.input_int (s, 0, valid_list=self.sharpen_dict.keys(), help_message="通过应用锐化滤镜来增强细节。")

        if self.sharpen_mode != 0:
            self.blursharpen_amount = np.clip ( io.input_int ("选择模糊/锐化量 [-100..100] (跳过:0) : ", 0), -100, 100 )

        s = """选择超级分辨率模式: \n"""
        for key in self.super_res_dict.keys():
            s += f"""({key}) {self.super_res_dict[key]}\n"""
        s += f"""帮助:? 默认: {list(self.super_res_dict.keys())[0]} : """
        self.super_resolution_mode = io.input_int (s, 0, valid_list=self.super_res_dict.keys(), help_message="通过应用超分辨率网络来增强细节。")

    def toggle_sharpen_mode(self):
        a = list( self.sharpen_dict.keys() )
        self.sharpen_mode = a[ (a.index(self.sharpen_mode)+1) % len(a) ]

    def add_blursharpen_amount(self, diff):
        self.blursharpen_amount = np.clip ( self.blursharpen_amount+diff, -100, 100)

    def toggle_super_resolution_mode(self):
        a = list( self.super_res_dict.keys() )
        self.super_resolution_mode = a[ (a.index(self.super_resolution_mode)+1) % len(a) ]

    #overridable
    def get_config(self):
        d = self.__dict__.copy()
        d.pop('type')
        return d
        return {'sharpen_mode':self.sharpen_mode,
                'blursharpen_amount':self.blursharpen_amount,
                'super_resolution_mode':self.super_resolution_mode
                }

    #overridable
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, ConverterConfig):
            return self.sharpen_mode == other.sharpen_mode and \
                   self.blursharpen_amount == other.blursharpen_amount and \
                   self.super_resolution_mode == other.super_resolution_mode

        return False

    #overridable
    def to_string(self, filename):
        r = ""
        r += f"锐化模式 : {self.sharpen_dict[self.sharpen_mode]}\n"
        r += f"锐化力度 [-100..100] : {self.blursharpen_amount}\n"
        r += f"超级分辨率模式 : {self.super_res_dict[self.super_resolution_mode]}\n"
        return r

mode_dict = {0:'original',
             1:'overlay',
             2:'hist-match',
             3:'seamless',
             4:'seamless-hist-match',
             5:'raw-rgb',
             6:'raw-rgb-mask',
             7:'raw-mask-only',
             8:'raw-predicted-only'}

mode_str_dict = {}

for key in mode_dict.keys():
    mode_str_dict[ mode_dict[key] ] = key

full_face_mask_mode_dict = {1:'learned',
                                    2:'dst',
                                    3:'FAN-prd',
                                    4:'FAN-dst',
                                    5:'FAN-prd*FAN-dst',
                                    6:'learned*FAN-prd*FAN-dst'}

half_face_mask_mode_dict = {1:'learned',
                                    2:'dst',
                                    4:'FAN-dst',
                                    7:'learned*FAN-dst'}

ctm_dict = { 0: "None", 1:"rct", 2:"lct", 3:"mkl", 4:"mkl-m", 5:"idt", 6:"idt-m", 7:"sot-m", 8:"mix-m" }
ctm_str_dict = {None:0, "rct":1, "lct":2, "mkl":3, "mkl-m":4, "idt":5, "idt-m":6, "sot-m":7, "mix-m":8 }

class ConverterConfigMasked(ConverterConfig):

    def __init__(self, face_type=FaceType.FULL,
                       default_mode = 'overlay',
                       clip_hborder_mask_per = 0,

                       mode='overlay',
                       masked_hist_match=True,
                       hist_match_threshold = 238,
                       mask_mode = 1,
                       erode_mask_modifier = 50,
                       blur_mask_modifier = 50,
                       motion_blur_power = 0,
                       output_face_scale = 0,
                       color_transfer_mode = ctm_str_dict['rct'],
                       image_denoise_power = 0,
                       bicubic_degrade_power = 0,
                       color_degrade_power = 0,
                       export_mask_alpha = False,
                       **kwargs
                       ):

        super().__init__(type=ConverterConfig.TYPE_MASKED, **kwargs)

        self.face_type = face_type
        if self.face_type not in [FaceType.HALF, FaceType.MID_FULL, FaceType.FULL ]:
            raise ValueError("ConverterConfigMasked不支持这种类型的面孔.")

        self.default_mode = default_mode
        self.clip_hborder_mask_per = clip_hborder_mask_per

        #default changeable params
        self.mode = mode
        self.masked_hist_match = masked_hist_match
        self.hist_match_threshold = hist_match_threshold
        self.mask_mode = mask_mode
        self.erode_mask_modifier = erode_mask_modifier
        self.blur_mask_modifier = blur_mask_modifier
        self.motion_blur_power = motion_blur_power
        self.output_face_scale = output_face_scale
        self.color_transfer_mode = color_transfer_mode
        self.image_denoise_power = image_denoise_power
        self.bicubic_degrade_power = bicubic_degrade_power
        self.color_degrade_power = color_degrade_power
        self.export_mask_alpha = export_mask_alpha

    def copy(self):
        return copy.copy(self)

    def set_mode (self, mode):
        self.mode = mode_dict.get (mode, self.default_mode)

    def toggle_masked_hist_match(self):
        if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
            self.masked_hist_match = not self.masked_hist_match

    def add_hist_match_threshold(self, diff):
        if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
            self.hist_match_threshold = np.clip ( self.hist_match_threshold+diff , 0, 255)

    def toggle_mask_mode(self):
        if self.face_type == FaceType.FULL:
            a = list( full_face_mask_mode_dict.keys() )
        else:
            a = list( half_face_mask_mode_dict.keys() )
        self.mask_mode = a[ (a.index(self.mask_mode)+1) % len(a) ]

    def add_erode_mask_modifier(self, diff):
        self.erode_mask_modifier = np.clip ( self.erode_mask_modifier+diff , -400, 400)

    def add_blur_mask_modifier(self, diff):
        self.blur_mask_modifier = np.clip ( self.blur_mask_modifier+diff , -400, 400)

    def add_motion_blur_power(self, diff):
        self.motion_blur_power = np.clip ( self.motion_blur_power+diff, 0, 100)

    def add_output_face_scale(self, diff):
        self.output_face_scale = np.clip ( self.output_face_scale+diff , -50, 50)

    def toggle_color_transfer_mode(self):
        self.color_transfer_mode = (self.color_transfer_mode+1) % ( max(ctm_dict.keys())+1 )

    def add_color_degrade_power(self, diff):
        self.color_degrade_power = np.clip ( self.color_degrade_power+diff , 0, 100)

    def add_image_denoise_power(self, diff):
        self.image_denoise_power = np.clip ( self.image_denoise_power+diff, 0, 500)

    def add_bicubic_degrade_power(self, diff):
        self.bicubic_degrade_power = np.clip ( self.bicubic_degrade_power+diff, 0, 100)

    def toggle_export_mask_alpha(self):
        self.export_mask_alpha = not self.export_mask_alpha

    def ask_settings(self):
        s = """选择模式: \n"""
        for key in mode_dict.keys():
            s += f"""({key}) {mode_dict[key]}\n"""
        s += f"""默认: { mode_str_dict.get(self.default_mode, 1)  } : """

        mode = io.input_int (s, mode_str_dict.get(self.default_mode, 1) )

        self.mode = mode_dict.get (mode, self.default_mode )

        if 'raw' not in self.mode:
            if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
                self.masked_hist_match = io.input_bool("蒙面组合匹配？(y/n 跳过:y):  ", True)

            if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip ( io.input_int("组合匹配阈值[0..255](跳过：255) :  ", 255), 0, 255)

        if self.face_type == FaceType.FULL:
            s = """选择遮罩模式: \n"""
            for key in full_face_mask_mode_dict.keys():
                s += f"""({key}) {full_face_mask_mode_dict[key]}\n"""
            s += f"""帮助:? 默认: 1 : """

            self.mask_mode = io.input_int (s, 1, valid_list=full_face_mask_mode_dict.keys(), help_message="（1）如果你学习面具，那么应该选择选项。（2）'dst'面具是来自dst对齐图像的原始摇晃面具。（3）'FAN-prd' - 使用来自预测面部的预训练FAN模型的超光滑面具。（4）'FAN-dst'  - 使用来自dst face的预训练FAN模型使用超光滑遮罩。（5）'FAN-prd * FAN-dst'或'learned * FAN-prd * FAN-dst' - 使用乘法遮罩。")
        else:
            s = """选择遮罩模式: \n"""
            for key in half_face_mask_mode_dict.keys():
                s += f"""({key}) {half_face_mask_mode_dict[key]}\n"""
            s += f"""帮助:? ,  默认: 1 : """
            self.mask_mode = io.input_int (s, 1, valid_list=half_face_mask_mode_dict.keys(), help_message="（1）如果你学习面具，那么应该选择选项。（2）'dst'面具是来自dst对齐图像的原始摇晃面具。")

        if 'raw' not in self.mode:
            self.erode_mask_modifier = np.clip ( io.input_int ("选择侵蚀面具修改器 [-400..400] (跳过:%d) : " % 0, 0), -400, 400)
            self.blur_mask_modifier =  np.clip ( io.input_int ("选择模糊遮罩修改器 [-400..400] (跳过:%d) : " % 0, 0), -400, 400)
            self.motion_blur_power = np.clip ( io.input_int ("选择运动模糊力度 [0..100] (跳过:%d) : " % (0), 0), 0, 100)

        self.output_face_scale = np.clip (io.input_int ("选择输出面部比例修改器 [-50..50] (跳过:0) : ", 0), -50, 50)

        if 'raw' not in self.mode:
            self.color_transfer_mode = io.input_str ( f"将颜色转移应用于预测的脸部吗？ 选择模式 ( {' / '.join ([str(x) for x in list(ctm_str_dict.keys())])} 跳过:None ) : ", None, ctm_str_dict.keys() )
            self.color_transfer_mode = ctm_str_dict[self.color_transfer_mode]

        super().ask_settings()

        if 'raw' not in self.mode:
            self.image_denoise_power = np.clip ( io.input_int ("选择图像降噪强度 [0..500] (跳过:%d) : " % (0), 0), 0, 500)
            self.bicubic_degrade_power = np.clip ( io.input_int ("选择图像双三次降低功率 [0..100] (跳过:%d) : " % (0), 0), 0, 100)
            self.color_degrade_power = np.clip (  io.input_int ("降低最终图像的色彩力度 [0..100] (跳过:0) : ", 0), 0, 100)
            self.export_mask_alpha = io.input_bool("用alpha通道导出png格式图片? (y/n 跳过:n) : ", False)

        io.log_info ("")

    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, ConverterConfigMasked):
            return super().__eq__(other) and \
                   self.mode == other.mode and \
                   self.masked_hist_match == other.masked_hist_match and \
                   self.hist_match_threshold == other.hist_match_threshold and \
                   self.mask_mode == other.mask_mode and \
                   self.erode_mask_modifier == other.erode_mask_modifier and \
                   self.blur_mask_modifier == other.blur_mask_modifier and \
                   self.motion_blur_power == other.motion_blur_power and \
                   self.output_face_scale == other.output_face_scale and \
                   self.color_transfer_mode == other.color_transfer_mode and \
                   self.image_denoise_power == other.image_denoise_power and \
                   self.bicubic_degrade_power == other.bicubic_degrade_power and \
                   self.color_degrade_power == other.color_degrade_power and \
                   self.export_mask_alpha == other.export_mask_alpha

        return False

    def to_string(self, filename):
        r = (
            f"""转换器配置 {filename}:\n"""
            f"""模式: {self.mode}\n"""
            )

        if self.mode == 'hist-match' or self.mode == 'hist-match-bw':
            r += f"""masked_hist_match: {self.masked_hist_match}\n"""

        if self.mode == 'hist-match' or self.mode == 'hist-match-bw' or self.mode == 'seamless-hist-match':
            r += f"""组合匹配阈值 [0..255] : {self.hist_match_threshold}\n"""

        if self.face_type == FaceType.FULL:
            r += f"""遮罩模式 : { full_face_mask_mode_dict[self.mask_mode] }\n"""
        else:
            r += f"""遮罩模式 : { half_face_mask_mode_dict[self.mask_mode] }\n"""

        if 'raw' not in self.mode:
            r += (f"""侵蚀面具修改器 [-400..400] : {self.erode_mask_modifier}\n"""
                  f"""模糊遮罩修改器 [-400..400] : {self.blur_mask_modifier}\n"""
                  f"""运动模糊力度 [0..100] : {self.motion_blur_power}\n""")

        r += f"""输出面部比例修改器 [-50..50] : {self.output_face_scale}\n"""

        if 'raw' not in self.mode:
            r += f"""颜色转移模式 : { ctm_dict[self.color_transfer_mode]}\n"""

        r += super().to_string(filename)

        if 'raw' not in self.mode:
            r += (f"""图像降噪强度: {self.image_denoise_power}\n"""
                  f"""双三次降低功率: {self.bicubic_degrade_power}\n"""
                  f"""降低图像色彩力度 : {self.color_degrade_power}\n"""
                  f"""alpha通道导出png格式图片蒙面 : {self.export_mask_alpha}\n""")

        r += "================"

        return r


class ConverterConfigFaceAvatar(ConverterConfig):

    def __init__(self, temporal_face_count=0,
                       add_source_image=False):
        super().__init__(type=ConverterConfig.TYPE_FACE_AVATAR)
        self.temporal_face_count = temporal_face_count

        #changeable params
        self.add_source_image = add_source_image

    def copy(self):
        return copy.copy(self)

    #override
    def ask_settings(self):
        self.add_source_image = io.input_bool("添加源图像? (y/n 帮助:? 跳过:n) : ", False, help_message="添加源图像进行比较。")
        super().ask_settings()

    def toggle_add_source_image(self):
        self.add_source_image = not self.add_source_image

    #override
    def __eq__(self, other):
        #check equality of changeable params

        if isinstance(other, ConverterConfigFaceAvatar):
            return super().__eq__(other) and \
                   self.add_source_image == other.add_source_image

        return False

    #override
    def to_string(self, filename):
        return (f"转换设置 {filename}:\n"
                f"添加源图像 : {self.add_source_image}\n") + \
                super().to_string(filename) + "================"

