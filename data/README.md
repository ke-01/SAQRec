Since the Commercial dataset is a proprietary industrial dataset, here we give the acquisition of the KuaiRand dataset. 

Please contact http://kuairand.com/ to get the KuaiRand dataset.

Then you can process it through construct_KuaiRand.ipynb.

After processing the provided data, the files in this folder should be organized as follows:
```bash
.
├── kuairand
│   ├── dataset
│   │   ├── kuairand_train.tsv
│   │   ├── kuairand_valid.tsv
│   │   └── kuairand_test.tsv
│   │   └── kuairand_satis.tsv
│   └── vocab
│       ├── kuairand_user_vocab.pickle
│       ├── kuairand_item_vocab.pickle
└── README.md
```

When training the satisfaction model, please pay attention to switching the training set to 'kuairan_satis.tsv'.


