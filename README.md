# 22

https://github.com/Silver-Wolf-han/22/tree/f35fef14af3b4671239e59f648ba048e20072fc6
is final version of pruning + HQQ + compile + Lora

### download and upzip
```=bash
curl -L -o commit-f35fef1.zip https://github.com/Silver-Wolf-han/22/archive/f35fef14af3b4671239e59f648ba048e20072fc6.zip
unzip commit-f35fef1.zip
```

### create env and install package
```=bash
cd 22-f35fef14af3b4671239e59f648ba048e20072fc6
python3 -m venv venv
pip install -r requirements.txt
```

### exe project
```=bash
huggingface-cli login
# Enter huggingface token
python result.py
```
### colab test
download `inference_test.ipynb` and run it on google colab
