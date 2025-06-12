# Granite Guardian Llama Stack External Provider (Proof of concept)

## Setup with Llama Stack

```
git clone https://github.com/bostrt/granite-guardian-shield
cd granite-guardian-shield

virtualenv v
source v/bin/activate
pip install -e .

export PYPI_VERSION=0.2.9
export CONTAINER_BINARY=podman
llama stack build --config build.yaml

# Make your own secrets.env file!
podman run --rm -it -v ./config.yaml:/app/run.yaml:ro,z -v ./providers.d/:/.llama/providers.d:ro,z --env-file secrets.env -p 8321:8321 my-llama-stack:0.2.9
```
