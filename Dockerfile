# Base image: Hugging Face Text Generation Inference server
FROM ghcr.io/huggingface/text-generation-inference:latest

# Model to serve by default (overridable at runtime with --model-id or MODEL_ID)
ENV MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct
# Where HF weights are cached inside the container (matches -v mount below)
ENV HUGGINGFACE_HUB_CACHE=/data
ENV HF_HUB_CACHE=/data
# TGI listens on port 80 inside the container by default
ENV PORT=80

# Documents that the container listens on 80 (does NOT publish it — still need -p at `docker run`)
EXPOSE 80

# Declares /data as cache dir for downloaded weights
VOLUME ["/data"]

# Default launch command. Override/extend with `docker run <image> --model-id ...`
ENTRYPOINT ["text-generation-launcher"]
CMD ["--model-id", "HuggingFaceTB/SmolLM2-360M-Instruct", "--port", "80"]
