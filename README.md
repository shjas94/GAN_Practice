# GAN Practice

## 📖 Reference
* [GAN](https://arxiv.org/pdf/1406.2661v1.pdf)
* [DCGAN](https://arxiv.org/pdf/1511.06434v2.pdf)

## 📃 ToDo List
* [WassersteinGAN](https://arxiv.org/pdf/1701.07875v3.pdf)
* [StyleGAN](https://arxiv.org/pdf/1812.04948v3.pdf)
* [StarGAN](https://arxiv.org/pdf/1711.09020v3.pdf)
* [SAGAN](https://arxiv.org/pdf/1805.08318v2.pdf)
* [TransGAN](https://arxiv.org/pdf/2102.07074v4.pdf)

## 🔧 Structure of code
```
.
├── Modules
│   ├── Discriminator.py
│   ├── Generator.py
│   ├── __init__.py
│   └── __pycache__
│       ├── Discriminator.cpython-37.pyc
│       ├── Generator.cpython-37.pyc
│       └── __init__.cpython-37.pyc
├── README.md
├── data
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   ├── dataloader.cpython-37.pyc
│   │   └── dataset.cpython-37.pyc
│   ├── dataloader.py
│   └── dataset.py
├── main.py
├── models
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-37.pyc
    │   └── utils.cpython-37.pyc
    └── utils.py
```


## 👨🏻‍💻 How to Train
```
$ python main.py
```