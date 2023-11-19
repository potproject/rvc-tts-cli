import os
import sys
import pyopenjtalk
import torch
import numpy
import uuid
import argparse
from scipy.io import wavfile

now_dir = os.getcwd()
sys.path.append(now_dir)
from infer.modules.vc.modules import VC
from configs.config import Config
from dotenv import load_dotenv

torch.manual_seed(114514)

load_dotenv()
config = Config()
vc = VC(config)

def main():
    # cli
    ## --work_dir <work_dir>
    ## --model <model>
    ## --sid <sid>
    ## --vc_transform <vc_transform>
    ## --f0method0 <f0method0>
    ## --index_rate <index_rate>
    ## --filter_radius <filter_radius>
    ## --resample_sr <resample_sr>
    ## --rms_mix_rate <rms_mix_rate>
    ## --protect <protect>

    work_dir = config.work_dir
    model = config.model
    sid = config.sid
    vc_transform = config.vc_transform
    f0method0 = config.f0method0
    index_rate = config.index_rate
    filter_radius = config.filter_radius
    resample_sr = config.resample_sr
    rms_mix_rate = config.rms_mix_rate
    protect = config.protect
    
    vc.get_vc(model)

    # interractive
    while True:
        text = input("rvc>> ")
        if text.strip() == "exit":
            print("Bye!")
            break
        # Change Model Command
        # rvc>> model: model.pth
        if text.strip().startswith("model "):
            model = text.strip().split("model ")[1]
            vc.get_vc(model)
            print("Model Changed!")
            continue
        # Change SID Command
        # rvc>> sid: 1
        if text.strip().startswith("sid "):
            sid = int(text.strip().split("sid ")[1])
            print("SID Changed!")
            continue
        
        # Text To Speech Command
        # rvc>> tts こんにちは output.wav
        if text.strip().startswith("tts "):
            split = text.strip().split(" ")
            speak_text = split[1]
            if speak_text == "":
                print("Please input text!")
                continue
            file_name_output = "output.wav"
            if len(split) > 2:
                file_name_output = split[2]
            if not file_name_output.endswith(".wav"):
                file_name_output += ".wav"

            x, sr = pyopenjtalk.tts(speak_text, run_marine=True)
            
            uniq = str(uuid.uuid4())
            file_name_input = uniq + ".wav"
            file_path_input = os.path.join(work_dir, file_name_input)
            file_path_output = os.path.join(work_dir, file_name_output)
            wavfile.write(file_path_input, sr, x.astype(numpy.int16))

            _, audio = vc.vc_single(sid, file_path_input, vc_transform, None, f0method0, "", "", index_rate, filter_radius, resample_sr, rms_mix_rate, protect)
            wavfile.write(file_path_output, audio[0], audio[1])
            #os.remove(file_path_input)
            print("saved to " + file_path_output)
            continue

        # Not found
        print("Command Not Found! Please try again!")
if __name__ == '__main__':
    main()