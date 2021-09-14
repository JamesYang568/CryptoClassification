# CryptoClassification Kit     
## Introduction
This is an effective kit for Crypto-text Classification which is created to classify the different ciphertexts encrypt 
by different encryption algorithms. In short, it handles the _**Classification task**_.      

Though it can only distinguish 4 different encryption algorithms, it has completely open up a new window to 
_**Cryptanalysis**_. By using this kit you can train a high-performance model or you just use the model that we had 
already provided.    

## How to use it
 1. run main in `backend.py`, see the routine and you will find how it works.    
 2. **Make Sure** there is any trained model in the folder `trained_model` if you run method `getCipherResult`.(This 
 means you want to use the model we provided.)
 3. **Make Sure** there is a folder named `self_model` if you run method `getTrainedModel` and then you need provide data
 in `./static/CiphertextFile/` .  
    
 The format of data to offer is:  _which **ciphertext file** for which **encryption algorithms** is in which **folder**_.     
 e.g. If you have 2 kinds of ciphertexts, one is encrypt by AES  and other is by 3DES, then put these ciphertexts into 
 corresponding folder named `./static/CiphertextFile/AES/` `./static/CiphertextFile/3DES`.
 (This means you want to train a model by yourself so you should load data.)    
 
 4. For 2 and 3 you can unzip the compressed files `self_model.zip`, `trained_model.zip`.    
 5. You really can understand the meanings simply through its name.

## Acknowledgment
 This work is completed by author and his team.    
 Thanks our professor XiangGuangLi.    
 Thanks my friends MoLi and Matthew.

## Contact
 yangjiaxiong@whut.edu.cn     
 or    
 2625398619@qq.com
