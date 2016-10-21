# Variational auto encoder: an autograd implementation

This implmentation in autograd follows the style of the numpy code.

Usage: download the [FreyFace](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl) 
data and make sure it's named freyface.pkl. Put that to the demo folder. Then run

```
python train_freyface.py -l num_layers -k num_samples --alpha alpha_value
```

See train_freyface.py for more options.
To see the max trick, add in one more option --backward_pass max.
