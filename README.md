# rvc-tts-cli
Based on [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) on TTS command line Tool

## Usage

* hubert_base.pt File: assets/hubert/hubert_base.pt
* rmvpe.pt File: assets/rmvpe/rmvpe.pt or assets/rmvpe/rmvpe.onnx
* Default Model File: assets/weights/default.pth

```sh
> python .\main.py --help
usage: main.py [-h] [--port PORT] [--pycmd PYCMD] [--colab] [--noparallel] [--noautoopen] [--dml] [--work_dir WORK_DIR] [--model MODEL]    [--sid SID]
               [--vc_transform VC_TRANSFORM] [--f0method0 F0METHOD0] [--index_rate INDEX_RATE] [--filter_radius FILTER_RADIUS] [--resample_sr RESAMPLE_SR]        
               [--rms_mix_rate RMS_MIX_RATE] [--protect PROTECT]
> python .\main.py --port 8888 --model default.pth --sid 0 --vc_transform 12
rvc>> tts おはようございます
saved to output.wav
```