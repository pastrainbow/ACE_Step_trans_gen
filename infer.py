import click
import os
import json
import torch
import torchaudio
from torchaudio.transforms import Resample

os.environ['HF_HOME'] = '/vol/bitbucket/al4624/cache/ace_step_cache/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/cache/ace_step_cache/xdg_cache_home'

from acestep.pipeline_ace_step import ACEStepPipeline
# from acestep.data_sampler import DataSampler

def sample_data(json_data):
    return (
        json_data["audio_duration"],
        json_data["prompt"],
        json_data["lyrics"],
        json_data["infer_step"],
        json_data["guidance_scale"],
        json_data["scheduler_type"],
        json_data["cfg_type"],
        json_data["omega_scale"],
        ", ".join(map(str, json_data["actual_seeds"])),
        json_data["guidance_interval"],
        json_data["guidance_interval_decay"],
        json_data["min_guidance_scale"],
        json_data["use_erg_tag"],
        json_data["use_erg_lyric"],
        json_data["use_erg_diffusion"],
        ", ".join(map(str, json_data["oss_steps"])),
        json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
        (
            json_data["guidance_scale_lyric"]
            if "guidance_scale_lyric" in json_data
            else 0.0
        ),
    )


def noise_gen_gaussian_stereo(range_factor, frame_count, device):
    mean = 0.0
    #portion of values in range = 1 - 1 / range_factor^2
    #value range is 1 here
    std = 1.0 / range_factor
    
    # Gaussian noise: create a random normal distribution that has the same size as the data to add noise to 
    # Genearte noise with same size as that of the data.
    ch_1 = torch.normal(mean=mean, std=std, size=(frame_count,), device=device)
    ch_2 = torch.normal(mean=mean, std=std, size=(frame_count,), device=device)
    return torch.stack((ch_1, ch_2))


def load_audio(path, sample_rate=44100):
    audio, sr = torchaudio.load(path)
    # Resample if needed
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)
    return audio

def prompts_concat(start_audio_path, end_audio_path, output_path, noise_duration, device, sample_rate=44100):
    range_factor = 4  # for gaussian noise generation

    start_audio_data = load_audio(start_audio_path, sample_rate).to(device)
    end_audio_data = load_audio(end_audio_path, sample_rate).to(device)
    
    noise_data = noise_gen_gaussian_stereo(
        range_factor,
        int(noise_duration * sample_rate),
        device,
    )
    concat_data = torch.cat((start_audio_data, noise_data, end_audio_data), dim=1)
    # print(concat_data.shape)
    torchaudio.save(output_path, concat_data, sample_rate)

    #since we had already loaded the start and end tracks, we can also get the repaint start and end times here
    start_audio_duration = start_audio_data.shape[-1] / sample_rate
    return (start_audio_duration, start_audio_duration + noise_duration)

    

@click.command()
@click.option(
    "--checkpoint_path", type=str, default="/vol/bitbucket/al4624/cache/ace_step_cache/model_cache", help="Path to the checkpoint directory"
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bfloat16")
@click.option(
    "--torch_compile", type=bool, default=False, help="Whether to use torch compile"
)
@click.option(
    "--cpu_offload", type=bool, default=False, help="Whether to use CPU offloading (only load current stage's model to GPU)"
)
@click.option(
    "--overlapped_decode", type=bool, default=False, help="Whether to use overlapped decoding (run dcae and vocoder using sliding windows)"
)
@click.option("--device_id", type=int, default=0, help="Device ID to use")
@click.option("--output_path", type=str, required=True, default=None, help="Path to save the output")
@click.option("--start_audio_path", type=str, required=True, default=None, help="Path to the starting audio clip")
@click.option("--end_audio_path", type=str, required=True, default=None, help="Path to the ending audio clip")
@click.option("--concat_audio_path", type=str, required=True, default=None, help="Path to the middle segment noised audio to save and load from")
@click.option("--repaint_variance", type=float, default=0.5, help="A float value between 0 and 1, determines how much the repaint section is noised, i.e. how varied the repainting will be")
@click.option("--gen_duration", type=float, default=10.0, help="The duration of the transition to be generated")
def main(checkpoint_path, bf16, torch_compile, cpu_offload, overlapped_decode, device_id, output_path, start_audio_path, end_audio_path, concat_audio_path, repaint_variance, gen_duration):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # data_sampler = DataSampler()

    # json_data = data_sampler.sample()
    with open("/vol/bitbucket/al4624/git_repo/ACE-Step/examples/default/input_params/trans_gen.json", "r") as f:
        json_data=json.load(f)
    json_data = sample_data(json_data)
    print(json_data)


    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode
    )
    print(model_demo)

    (
        audio_duration,
        prompt,
        lyrics,
        infer_step,
        guidance_scale,
        scheduler_type,
        cfg_type,
        omega_scale,
        manual_seeds,
        guidance_interval,
        guidance_interval_decay,
        min_guidance_scale,
        use_erg_tag,
        use_erg_lyric,
        use_erg_diffusion,
        oss_steps,
        guidance_scale_text,
        guidance_scale_lyric,
    ) = json_data

    #create middle segment noised audio
    #Run on cpu since the concat process should be quite low cost 
    device = torch.device("cpu")
    repaint_start, repaint_end = prompts_concat(start_audio_path, end_audio_path, concat_audio_path, gen_duration, device)
    src_audio_path=concat_audio_path
    # output_path="/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/mixture_audio_noised/078303.denoised.wav"

    #repainting inference
    model_demo(
        format="wav",
        audio_duration=audio_duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=omega_scale,
        manual_seeds=manual_seeds,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
        min_guidance_scale=min_guidance_scale,
        use_erg_tag=use_erg_tag,
        use_erg_lyric=use_erg_lyric,
        use_erg_diffusion=use_erg_diffusion,
        oss_steps=oss_steps,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        retake_seeds=None,
        retake_variance=repaint_variance,
        task="repaint",
        repaint_start=repaint_start,
        repaint_end=repaint_end,
        src_audio_path=src_audio_path,
        lora_name_or_path="none",
        lora_weight=1,
        save_path=output_path,

    )


if __name__ == "__main__":
    main()
