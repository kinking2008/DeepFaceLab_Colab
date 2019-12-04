import subprocess
import numpy as np
import ffmpeg
from pathlib import Path
from utils import Path_utils
from interact import interact as io

def extract_video(input_file, output_dir, output_ext=None, fps=None):
    input_file_path = Path(input_file)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(exist_ok=True)


    if input_file_path.suffix == '.*':
        input_file_path = Path_utils.get_first_file_by_stem (input_file_path.parent, input_file_path.stem)
    else:
        if not input_file_path.exists():
            input_file_path = None

    if input_file_path is None:
        io.log_err("输入目录没有文件")
        return

    if fps is None:
        fps = io.input_int ("提取每秒视频 FPS ( 帮助:? 跳过:按原视频FPS ) : ", 0, help_message="提取每秒视频的帧数。")

    if output_ext is None:
        output_ext = io.input_str ("输出图像格式? ( jpg png 帮助:? 跳过:png ) : ", "png", ["png","jpg"], help_message="png格式是无损的，但对于HDD来说，提取速度要慢10倍，需要比jpg格式多10倍的磁盘空间。")

    for filename in Path_utils.get_image_paths (output_path, ['.'+output_ext]):
        Path(filename).unlink()

    job = ffmpeg.input(str(input_file_path))

    kwargs = {'pix_fmt': 'rgb24'}
    if fps != 0:
        kwargs.update ({'r':str(fps)})

    if output_ext == 'jpg':
        kwargs.update ({'q:v':'2'}) #highest quality for jpg

    job = job.output( str (output_path / ('%5d.'+output_ext)), **kwargs )

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg失败，工作命令行:" + str(job.compile()) )

def cut_video ( input_file, from_time=None, to_time=None, audio_track_id=None, bitrate=None):
    input_file_path = Path(input_file)
    if input_file_path is None:
        io.log_err("输入目录没有文件")
        return

    output_file_path = input_file_path.parent / (input_file_path.stem + "_cut" + input_file_path.suffix)

    if from_time is None:
        from_time = io.input_str ("开始时间 (跳过: 00:00:00.000) : ", "00:00:00.000")

    if to_time is None:
        to_time = io.input_str ("结束时间 (跳过: 00:00:00.000) : ", "00:00:00.000")

    if audio_track_id is None:
        audio_track_id = io.input_int ("指定音轨ID。( 跳过:0 ) : ", 0)

    if bitrate is None:
        bitrate = max (1, io.input_int ("输出文件的比特率每秒多少MB? (默认:25) : ", 25) )

    kwargs = {"c:v": "libx264",
              "b:v": "%dM" %(bitrate),
              "pix_fmt": "yuv420p",
             }

    job = ffmpeg.input(str(input_file_path), ss=from_time, to=to_time)

    job_v = job['v:0']
    job_a = job['a:' + str(audio_track_id) + '?' ]

    job = ffmpeg.output(job_v, job_a, str(output_file_path), **kwargs).overwrite_output()

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg失败，工作命令行:" + str(job.compile()) )

def denoise_image_sequence( input_dir, ext=None, factor=None ):
    input_path = Path(input_dir)

    if not input_path.exists():
        io.log_err("input_dir not found.")
        return

    if ext is None:
        ext = io.input_str ("输入图像格式（扩展名：jpg/png）? ( 默认:png ) : ", "png")

    if factor is None:
        factor = np.clip ( io.input_int ("降噪因子? (1-20 默认:5) : ", 5), 1, 20 )

    kwargs = {}
    if ext == 'jpg':
        kwargs.update ({'q:v':'2'})
        
    job = ( ffmpeg
            .input(str ( input_path / ('%5d.'+ext) ) )
            .filter("hqdn3d", factor, factor, 5,5)
            .output(str ( input_path / ('%5d.'+ext) ), **kwargs )
           )

    try:
        job = job.run()
    except:
        io.log_err ("ffmpeg失败，工作命令行:" + str(job.compile()) )

def video_from_sequence( input_dir, output_file, reference_file=None, ext=None, fps=None, bitrate=None, lossless=None ):
    input_path = Path(input_dir)
    output_file_path = Path(output_file)
    reference_file_path = Path(reference_file) if reference_file is not None else None

    if not input_path.exists():
        io.log_err("input_dir not found.")
        return

    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        return

    out_ext = output_file_path.suffix

    if ext is None:
        ext = io.input_str ("输入图像格式（扩展名：jpg/png）? ( 默认:png ) : ", "png")

    if lossless is None:
        lossless = io.input_bool ("使用无损编解码器? ( 默认:no ) : ", False)

    video_id = None
    audio_id = None
    ref_in_a = None
    if reference_file_path is not None:
        if reference_file_path.suffix == '.*':
            reference_file_path = Path_utils.get_first_file_by_stem (reference_file_path.parent, reference_file_path.stem)
        else:
            if not reference_file_path.exists():
                reference_file_path = None

        if reference_file_path is None:
            io.log_err("参照目录没有文件.")
            return

        #probing reference file
        probe = ffmpeg.probe (str(reference_file_path))

        #getting first video and audio streams id with fps
        for stream in probe['streams']:
            if video_id is None and stream['codec_type'] == 'video':
                video_id = stream['index']
                fps = stream['r_frame_rate']

            if audio_id is None and stream['codec_type'] == 'audio':
                audio_id = stream['index']

        if audio_id is not None:
            #has audio track
            ref_in_a = ffmpeg.input (str(reference_file_path))[str(audio_id)]

    if fps is None:
        #if fps not specified and not overwritten by reference-file
        fps = max (1, io.input_int ("FPS ? (默认:25) : ", 25) )

    if not lossless and bitrate is None:
        bitrate = max (1, io.input_int ("输出文件的比特率每秒多少MB? (默认:16) : ", 16) )

    input_image_paths = Path_utils.get_image_paths(input_path)

    i_in = ffmpeg.input('pipe:', format='image2pipe', r=fps)
    
    output_args = [i_in]

    if ref_in_a is not None:
        output_args += [ref_in_a]

    output_args += [str (output_file_path)]

    output_kwargs = {}

    if lossless:
        output_kwargs.update ({"c:v": "png"
                              })
    else:
        output_kwargs.update ({"c:v": "libx264",
                               "b:v": "%dM" %(bitrate),
                               "pix_fmt": "yuv420p",
                              })

    output_kwargs.update ({"c:a": "aac",
                           "b:a": "192k",
                           "ar" : "48000"
                          })

    job = ( ffmpeg.output(*output_args, **output_kwargs).overwrite_output() )

    try:        
        job_run = job.run_async(pipe_stdin=True)
        
        for image_path in input_image_paths:
            with open (image_path, "rb") as f:
                image_bytes = f.read()            
                job_run.stdin.write (image_bytes)
        
        job_run.stdin.close()
        job_run.wait()
    except:
        io.log_err ("ffmpeg失败，工作命令行:" + str(job.compile()) )
