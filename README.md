# ancient-chinese-machine-translation
Project repo for course csci2470.

## Installation
Python3.6+ needed. 

### Required packages:
```
regex==2018.1.10
terminaltables==3.1.0
torch==1.3.0
numpy==1.14.0
tensorboardX==1.9
```
Easily, you can install all requirement with:

```
pip3 install -r requirements.txt
```

Also, you may need to use a useful tokenization tool [HanLP](https://github.com/hankcs/HanLP/tree/doc-zh) to tokenize the chinese words:
```
pip install hanlp
```