# DeCART Deep Learning for NLP

For this course we are accessing a machine that has 4 GPUs (0-3). By default each TensorFlow session will grab all the memory on all machines. In order for us to share the GPUs, we will have have to limit our access to particular GPU(s) and the memory we use on those GPUs. Consequently, all of our notebooks will need to have the following code snippet:

```Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras import backend as K

cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
cfg.gpu_options.per_process_gpu_memory_fraction=0.333
K.set_session(K.tf.Session(config=cfg))
```

Where each user/group will need to have `CUDA_VISIBLE_DEVICES`  as 0, 1, 2, or 3.

