Undergoing Restructuring at the moment! Thank you :)

1. Clone this repo.
   
2. Run in terminal:
   
   git clone https://github.com/hannamw/EAP-positional.git

   apt-get install -y git-lfs

   sudo apt-get install -y python3-dev graphviz libgraphviz-dev pkg-config

3. Install requirements.

4. Run for preliminary Bias analysis:

   python main.py 'sen' 'model_name' --HF_token 'HF_token'
   
5. Run for Edge Attribution:

   python main.py 'sen' 'model_name' 'metric' --HF_token 'HF_token'

sen = DSS1 or DSS2 (DSS1: "<Nationality> people are so", DSS2: "As expected, <Nationality> people are so") (where <Nationality> is replaced by the nationalities from the dataset)

model_name = gpt2, gpt2-large, llama2-7B, Gemma, Qwen

metric = M1 or M2 (M1=Sum of Probability Differences between next tokens of positive and negative sentiments, M2=Sum of Probabilities of next tokens with positive sentiment)

Paper link: https://openreview.net/forum?id=EpQ2CBJTjD

cite:

@article{

bashir2025dissecting,

title={Dissecting Bias in {LLM}s: A Mechanistic Interpretability Perspective},

author={Zubair Bashir and Bhavik Chandna and Procheta Sen},

journal={Transactions on Machine Learning Research},

issn={2835-8856},

year={2025},

url={https://openreview.net/forum?id=EpQ2CBJTjD},

note={}

}


