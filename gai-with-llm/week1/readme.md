To check the original jupyter notebook, you can directly run it if you have jupyter env in your local

## Steps to run week 1 lab in local
1. Used python 3.11.5
2. Create a virtual environment
``` 
python3 -m venv env/llm-lab
source env/llm-lab/bin/activate

pip3 install --upgrade pip
```
3. Install the dependencies
```
pip3 install --upgrade pip
pip3 install tensorflow==2.13.0 keras==2.13.1
pip3 install --no-deps torch==2.0.0 torchdata==0.6.0 --quiet
pip3 install -U \
    datasets==2.17.0 \
    transformers==4.27.2 \
    evaluate==0.4.0 \
    rouge_score==0.1.2 \
    peft==0.3.0 --quiet

```

4. Now run lab.py
```
python3 lab.py
```