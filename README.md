# Granite Guardian Llama Stack External Provider (Proof of concept)

## Setup with Llama Stack

```
git clone https://github.com/bostrt/granite-guardian-shield
cd granite-guardian-shield

virtualenv v
source v/bin/activate
pip install -e .

llama stack build --config build.yaml

# Create your own secrets.env
cp secrets.env.example secrets.env
export $(cat secrets.env)

# Edit run.yaml to your liking

# Run it
llama stack run --image-type venv ./run.yaml
```
