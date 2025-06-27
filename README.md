# Granite Guardian Llama Stack External Provider (Proof of concept)

## Setup with Llama Stack

```
git clone https://github.com/bostrt/granite-guardian-shield
cd granite-guardian-shield

virtualenv v
source v/bin/activate
pip install -e .

llama stack build --config build.yaml

# Create your own config.yaml or try setting each of the environment vars
llama stack run --image-type venv \
--env VLLM_URL=https://example.com/v1 \
--env VLLM_API_TOKEN=SECRET-TOKEN \
--env INFERENCE_MODEL=my-model \
--env GRANITE_API_KEY=SECRET-TOKEN \
--env GRANITE_BASE_URL=https://example2.com/v1 \
./config.yaml
```
