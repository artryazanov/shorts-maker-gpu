# Troubleshooting

If you run into issues using Shorts Maker GPU, refer to these common problems and their solutions.

!!! failure "Segmentation Fault during Docker Build"
    **Error:** `"internal compiler error: Segmentation fault"` during `docker build`.
    
    This typically occurs due to an Out-Of-Memory (OOM) error when Docker attempts to compile heavy C++/CUDA libraries (like VPF) using all available CPU cores. 
    
    **Solution:** Limit the number of CPU cores used during the build process:
    ```bash
    docker build --cpuset-cpus="0,1" -t shorts-maker .
    ```
    *(Alternatively, you can increase the RAM limit for Docker/WSL2 in your system settings).*

!!! warning "GPU or Torch not found"
    **Error:** `"Torch not installed"` / `"CUDA not available"`
    
    **Solution:** Ensure you are running inside the Docker container with `--gpus all` or have the correct CUDA toolkit installed locally matching your PyTorch build.

!!! bug "NVENC Error"
    **Error:** Rendering crashes when trying to encode video.
    
    **Solution:** If `h264_nvenc` fails, the script attempts to fall back to software encoding (`libx264`). Check if your GPU natively supports NVENC (some entry-level or mobile GPUs don't) and verify that your NVIDIA drivers are up to date.

!!! failure "WSL Crash / OOM during Docker Run"
    **Error:** `"WSL integration with distro unexpectedly stopped"` or sudden container crashes during `docker run`.
    
    This happens when video processing exhausts the memory (RAM/VRAM) allocated to the WSL2 virtual machine.
    
    **Solution:** Limit the number of CPU cores the container uses at runtime to throttle parallel processing, using the `--cpus` flag:
    ```bash
    docker run --rm --gpus all --cpus="4.0" -v $(pwd)/gameplay:/app/gameplay -v $(pwd)/generated:/app/generated --env-file .env shorts-maker
    ```
