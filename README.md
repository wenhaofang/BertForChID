### Introduction

This is a repository for Idiom NER and Idiom Cloze

referring to [ChID: A Large-scale Chinese IDiom Dataset for Cloze Test](https://aclanthology.org/P19-1075/).



### Data Process

1. Download ChID dataset into `data/chid` folder from [here](https://drive.google.com/drive/folders/1qdcMgCuK9d93vLVYJRvaSLunHUsGf50u)

   including `train_data.txt`, `dev_data.txt`, `test_data.txt` files

2. Download bert-base-chinese model into `data/bert` folder from [here](https://huggingface.co/bert-base-chinese/tree/main)

   including `config.json`, `vocab.txt`, `pytorch_model.bin` files



### Main Process

* Task One: Idiom NER

```shell
python main1.py --name NER
```

* Task Two: Idiom Cloze

```shell
python main2.py --name Cloze
```

You can modify the configuration through command line parameters or `parser.py`



