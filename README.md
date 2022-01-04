# GAN Practice

## 📖 Reference
* [GAN](https://arxiv.org/pdf/1406.2661v1.pdf)
* [DCGAN](https://arxiv.org/pdf/1511.06434v2.pdf)
* [WassersteinGAN](https://arxiv.org/pdf/1701.07875v3.pdf)
* [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028v3.pdf)
## 📃 ToDo List
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
│   └── __init__.py
├── README.md
├── data
│   ├── __init__.py
│   ├── dataloader.py
│   └── dataset.py
├── main.py
├── models
└── utils
    ├── __init__.py
    ├── losses.py
    ├── optimizers.py
    └── utils.py
```


## 👨🏻‍💻 How to Train
### DCGAN
```
$ python main.py
```

### WGAN
```
$ python main.py --model_name="WGAN"
```

### WGAN-GP
```
$ python main.py --model_name="WGAN-GP"
```