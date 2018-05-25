from keras.models import load_model
import tensorflowjs as tfjs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inpath", help="path of the model to convert to tfjs")
parser.add_argument("--outpath", help="destination path")
args = parser.parse_args()
path_in = args.inpath
path_out = args.outpath
print(args, path_in, path_out)

mod = load_model(path_in)
tfjs.converters.save_keras_model(mod, path_out)