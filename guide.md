git clone https://github.com/hannamw/EAP-positional.git
apt-get install -y git-lfs 
sudo apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config

python Bias_Code/Demographic_Bias/ModelBias.py DSS2 gpt2 --HF_token "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"

python Bias_Code/Demographic_Bias/main.py DSS2 gpt2 M1 --HF_token "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"
