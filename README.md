# mobileKGQA

## Environment setup
```
# ollama install # 0.5.13 version
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.5.13 sh

# venv setup
conda create -n mobileKGQA python=3.9
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.45.2
pip install bitsandbytes==0.43.3
pip install accelerate==0.34.2
pip install bitmath
pip install langchain_community==0.3.0
pip install datasets==3.0.0
pip install scikit-learn==1.5.1
```

## preprocessing
Below code is very time-consuming. We highly recommend to download preprocessed data from [here](https://drive.google.com/drive/folders/1Bjje9LU6KO-1RVc29qTKfdV3cjZW_FUD?usp=sharing). After creating the ./data directory and extracting the downloaded files, you can skip the preprocessing step.
```
sh ./scripts/ollama.sh # run llama3.1 ollama server (select GPU, port you want)
python ./preprocess/domain_split.py
python ./preprocess/save_last_hidden_state.py --domain_list total domain1 domain2 domain3
python ./preprocess/gen_questions.py
```

## question generation
```
python ./preprocess
```