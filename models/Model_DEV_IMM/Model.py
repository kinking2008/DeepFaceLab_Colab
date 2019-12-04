import colorsys
from functools import partial

import cv2
import numpy as np

from facelib import FaceType
from interact import interact as io
from mathlib import get_power_of_two
from models import ModelBase
from nnlib import IMM, nnlib
from samplelib import *


class IMMModel(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)

    #override
    def onInitializeOptions(self, is_first_run, ask_override):
        default_face_type = 'f'
        if is_first_run:
            self.options['resolution'] = io.input_int("分辨率 ( 128,224 帮助 :? 跳过:128) : ", 128, [128,224])
        else:
            self.options['resolution'] = self.options.get('resolution', 128)

        if (is_first_run or ask_override) and 'tensorflow' in self.device_config.backend:
            def_optimizer_mode = self.options.get('optimizer_mode', 1)
            self.options['optimizer_mode'] = io.input_int ("优化模式? ( 1,2,3 帮助:? 跳过:%d) : " % (def_optimizer_mode), def_optimizer_mode, help_message="1 - 没有变化。2 - 允许您训练x2更大的网络消耗内存。3 - 允许你训练x3更大的网络消耗大量的内存和更慢，取决于CPU的功率。")
        else:
            self.options['optimizer_mode'] = self.options.get('optimizer_mode', 1)
            
    #override
    def onInitialize(self, batch_size=-1, **in_options):
        exec(nnlib.code_import_all, locals(), globals())
        self.set_vram_batch_requirements({4:50})

        resolution = self.options['resolution']
        face_type = FaceType.FULL 
        
        self.model = IMM ( initialize_weights=self.is_first_run(),     
                            is_training=self.is_training_mode,
                            tf_cpu_mode=self.options['optimizer_mode']-1
                           )
                             
        if not self.is_first_run():
            self.load_weights_safe(self.model.get_model_filename_list())       
            
        if self.is_training_mode:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL
            
            output_sample_types=[ {'types': (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':128},
                                  {'types': (t.IMG_SOURCE, face_type, t.MODE_BGR), 'resolution':128},
                                ]

            self.set_training_data_generators ([
                        SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types ),

                        SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size,
                            sample_process_options=SampleProcessor.Options(random_flip=True),
                            output_sample_types=output_sample_types ),
                    ])

    #override
    def get_model_filename_list(self):
        return self.model.get_model_filename_list()

    #override
    def onSave(self):
        self.save_weights_safe(self.model.get_model_filename_list())

    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        warped_src, target_src = generators_samples[0]
        warped_dst, target_dst = generators_samples[1]
        
        G_loss = self.model.train( warped_src, target_src )

        return ( ('G_loss', G_loss), )

    #override
    def onGetPreview(self, generators_samples):
        view_samples = 1
        
        warped_src  = generators_samples[0][0][0:view_samples]
        target_src  = generators_samples[0][1][0:view_samples]
        warped_dst  = generators_samples[1][0][0:view_samples]
        target_dst  = generators_samples[1][1][0:view_samples]

        lines_train = []
        lines_test = []
        
        src, src_maps = self.model.convert  ([ warped_src, target_src  ] )
        dst, dst_maps = self.model.convert  ([ warped_dst, target_dst  ] )

        

            


        for i in range(view_samples):
            
            
            map = np.zeros( (128,128,3), dtype=np.float32 )
            for n in range(68):
                x,y = src_maps[i, n]
                color = colorsys.hsv_to_rgb ( n * (1.0/68), 1.0, 1.0 )
                cv2.circle (map, ( int(x*128), int(y*128) ), 1, color, lineType=cv2.LINE_AA )
                
            line_train = [ warped_src[i], target_src[i], src[i], map ]
            
            #import code
            #code.interact(local=dict(globals(), **locals()))
            
            lines_train += [ np.concatenate([ x for x in line_train], axis=1) ]
            
            line_test = [ warped_dst[i], target_dst[i], dst[i] ]
            lines_test  += [ np.concatenate([ x for x in line_test ], axis=1) ]

            

        lines_train = np.concatenate ( lines_train, axis=0 )
        #lines_test = np.concatenate ( lines_test, axis=0 )
        ('TEST', lines_test)
        r =  [ ('TRAIN', lines_train ),  ]
        
        return r

    def predictor_func (self, face=None, dummy_predict=False):
        if dummy_predict:
            self.model.convert ([ np.zeros ( (1, self.options['resolution'], self.options['resolution'], 3), dtype=np.float32 ), self.average_class_code ])
        else:
            bgr, = self.model.convert ([  face[np.newaxis,...]*2-1, self.average_class_code  ])
            return bgr[0] / 2 + 0.5

    #override
    def get_ConverterConfig(self):
        face_type = FaceType.FULL

        import converters
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), converters.ConverterConfigMasked(face_type=face_type,
                                     default_mode = 1,
                                     clip_hborder_mask_per=0.0625 if (face_type == FaceType.FULL) else 0,
                                    )


Model = IMMModel
