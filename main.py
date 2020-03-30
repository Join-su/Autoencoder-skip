import numpy as np
import math
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import tensorflow as tf
import matplotlib.colors as c
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from subprocess import call

from keras.models import Model
from keras.layers import Lambda,Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape, Flatten, Add, concatenate
from keras import optimizers, Sequential
from keras.models import load_model
from utils import *

from keras.datasets import mnist

BatchSize = 32
Epochs = 25

skip = 0
deno = 1
test = 0
nom = 0

save_path = './save_model/'
graph_save_path = './graph/'
if test == 1:
    en_save = 'test_{}_encoder'.format(Epochs)
    auto_save = 'test_{}_autoencoder'.format(Epochs)
    graph_name = 'test_{}_graph'.format(Epochs)
    img_name = 'test_{}_img'.format(Epochs)
elif skip == 1:

    en_save = 'skip_{}_encoder'.format(Epochs)
    auto_save = 'skip_{}_autoencoder'.format(Epochs)
    graph_name = 'skip_{}_graph'.format(Epochs)
    img_name = 'skip_{}_img'.format(Epochs)
    '''
    en_save = 'skip_tset_{}_encoder'.format(Epochs)
    auto_save = 'skip_tset_{}_autoencoder'.format(Epochs)
    graph_name = 'skip_tset_{}_graph'.format(Epochs)
    img_name = 'skip_tset_{}_img'.format(Epochs)
    '''

elif deno == 1:
    en_save = 'deno_{}_encoder'.format(Epochs)
    auto_save = 'deno_{}_autoencoder'.format(Epochs)
    graph_name = 'deno_{}_graph'.format(Epochs)
    img_name = 'deno_{}_img'.format(Epochs)
else :
    en_save = 'nom_{}_encoder'.format(Epochs)
    auto_save = 'nom_{}_autoencoder'.format(Epochs)
    graph_name = 'nom_{}_graph'.format(Epochs)
    img_name = 'nom_{}_img'.format(Epochs)

def main():

    # Parameters: whether to train a new neural network (or load already-trained from disk),
    # how many to train, how many to predict (or load from disk) and visualize (dimension? whether to build 3d .gif?)
    train_new = False
    n_train = 60000
    predict_new = False
    n_predict = 6000
    vis_dim = 2
    build_anim = False


    # Load MNIST dataset.
    x_train, y_train, x_test, y_test = import_format_data()

    x_train = x_train.astype(np.float)
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.astype(np.float)
    x_test = x_test.reshape(10000, 28, 28, 1)


    # for ind in range(0,2) :
    #     if ind == 1:
    #         skip = 1
    #         deno = 0
    #         en_save = 'skip_{}_encoder'.format(Epochs)
    #         auto_save = 'skip_{}_autoencoder'.format(Epochs)
    #     else :
    #         skip = 0
    #         deno = 1
    #         en_save = 'deno_{}_encoder'.format(Epochs)
    #         auto_save = 'deno_{}_autoencoder'.format(Epochs)

    # Build and fit autoencoder
    if train_new:
        #autoencoder, encoder = build_autoencoder((x_train.shape[1],), encoding_dim=30)
        autoencoder, encoder = build_autoencoder((x_train.shape[1],), encoding_dim=30)
        autoencoder.compile(optimizer=optimizers.Adadelta(), loss='mean_squared_error')

        autoencoder.summary()
        autoencoder.fit(x_train[:n_train], x_train[:n_train], epochs=Epochs, batch_size=BatchSize)
        autoencoder.save(save_path + auto_save)
        encoder.save(save_path + en_save)
    else:
        encoder = load_model(save_path + en_save)
        autoencoder = load_model(save_path + auto_save,custom_objects={'tf': tf})

        autoencoder_last_layer = autoencoder.layers[-1].get_weights()
        #conv_out = autoencoder.layers[-4].output ## skip part conv2d_8
        #conv_out = autoencoder.layers[-5].output ## skip part conv2d_7

        conv_out = autoencoder.layers[-3].output ## denoising part conv2d_6

        #conv2_8_output = Add(name='test_add_1')([conv2_7_output,conv2_8_output])

        x = UpSampling2D((2, 2), name= 'test_UpSampling2D_1')(conv_out)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='test_conv2d_1')(x)
        test_model = Model(autoencoder.inputs, x)
        test_model.compile(optimizer=optimizers.Adadelta(), loss='mean_squared_error')
        test_model.summary()
        print(np.shape(autoencoder_last_layer))
        #autoencoder_last_layer = Lambda(lambda x: tf.reshape(x, shape=[3,3,16,1]))(autoencoder_last_layer)
        #test_model.set_weights(autoencoder_last_layer)



    #decoded_imgs = autoencoder.predict(x_test)
    decoded_imgs = test_model.predict(x_test)


    '''
    import sys

    for layer in autoencoder.layers:
        config = layer.get_config()
        weights = layer.get_weights()
        sys.stdout = open('output.txt','a')
        print(config)
        print(weights)
    '''


    # Encode a number of MNIST digits, then perform t-SNE dim-reduction.
    if predict_new:
        x_train_predict = encoder.predict(x_test[:n_predict])
        #x_train_predict = x_test[:n_predict]
        #x_train_predict = np.reshape(x_train_predict,(n_predict,784))
        print(np.shape(x_train_predict))


        print("Performing t-SNE dimensionality reduction...")
        x_train_encoded = TSNE(n_components=vis_dim).fit_transform(x_train_predict)
        np.save(save_path + '%sx_%sdim_tnse_%s.npy' % (n_predict, vis_dim, n_train), x_train_encoded)
        print("Done.")
    else:
        x_train_encoded = np.load(str(n_predict) + 'x_' + str(vis_dim) + 'dim_tnse_' + str(n_train) + '.npy')

    vis_data(x_train_encoded, y_test, vis_dim, n_predict, n_train, build_anim, 1)  # 정답


    if vis_dim == 2 :
        model = KMeans(init="k-means++", n_clusters=10, random_state=0)
        model.fit(x_train_encoded)
        y_pred = model.labels_

        # count = 0
        #
        # for j in range(len(y_pred)) :
        #     if y_pred[j] == y_test[j] :
        #         count += 1
        #
        # print('정확도 : ', (count/len(y_pred)))

        vis_data(x_train_encoded, y_pred, vis_dim, n_predict, n_train, build_anim, 2) # Clusturing

    if test == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/test_{}.png'.format("skip_connections", Epochs))
    elif skip == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/skip_{}.png'.format("skip_connections", Epochs))
        #save_images(decoded_imgs[:100], image_manifold_size(100), './{}/skip_test_{}.png'.format("skip_connections", Epochs))
        print('save file')
    elif deno == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/part_deno_{}.png'.format("skip_connections", Epochs))
    else :
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/nom_{}.png'.format("skip_connections", Epochs))

        #x_test = decoded_imgs

def import_format_data():
    # Get dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Turn [0,255] values in (N, A, B) array into [0,1] values in (N, A*B) flattened arrays
    x_train = x_train.astype('float64') / 255.0
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
    x_test = x_test.astype('float64') / 255.0
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    if deno == 1 or nom == 1:
        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)

        #return x_train_noisy, y_train, x_test_noisy, y_test
        return x_train_noisy, y_train, x_test, y_test

    return x_train, y_train, x_test, y_test



def build_autoencoder(input_shape, encoding_dim):
    # Activation function: selu for SNN's: https://arxiv.org/pdf/1706.02515.pdf
    encoding_activation = 'selu'
    decoding_activation = 'selu'

    # Preliminary parameters
    #inputs = Input(shape=input_shape) #Tensor("input_1:0", shape=(?, 784), dtype=float32)
    #feat_dim = input_shape[0] #784

    inputs = Input(shape=(28,28,1,))


    # Encoding layers: successive smaller layers, then a batch normalization layer.
    # Conv1 #
    encoding = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    #z2 = encoding

    # Conv2 #
    encoding = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    z2 = encoding

    # Conv3 #
    encoding = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    z1 = encoding
    #print(encoding)
    # Reshape((sequence_length, embedding_dimension, 1), input_shape = (sequence_length, embedding_dimension))
    flatten_layer = Flatten()  # instantiate the layer
    encoding = flatten_layer(encoding)  # call it on the given tensor
    feat_dim = 4*4*8

    encoding = Dense(feat_dim, activation=encoding_activation, kernel_initializer='lecun_normal')(encoding)
    encoding = Dense(int(feat_dim/2), activation=encoding_activation)(encoding)
    #encoding = Dense(int(feat_dim/4), activation=encoding_activation)(encoding)
    encoding = Dense(encoding_dim, activation=encoding_activation)(encoding)

    # Decoding layers for reconstruction
    #decoding = Dense(int(feat_dim/4), activation=decoding_activation)(encoding)
    decoding = Dense(int(feat_dim/2), activation=decoding_activation)(encoding)
    decoding = Dense(feat_dim, activation=decoding_activation)(decoding)
    print(decoding)

    decoding = Lambda(lambda x: tf.reshape(x, shape=[-1,4,4,8]))(decoding)
    #decoding = Reshape((-1,7,7,8),input_shape = (BatchSize,7*7*8))(decoding)  # (,392)
    print('1 : ',decoding)

    if skip == 1 :
        z1 = Conv2D(8, (3, 3), activation='relu', padding='same')(z1)
        #z1 = Conv2D(8, (1, 1), activation='relu', padding='same')(z1)
        decoding = Add()([decoding,z1])
        #joinedTensor = Add()([x,complement])

    print('2 :',decoding)

    decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding) # 4*4*8
    decoding = UpSampling2D((2, 2))(decoding) #8*8*8

    if skip == 1:
        z2 = UpSampling2D((2, 2))(z2)
        z2 = Conv2DTranspose(8, (3, 3), activation='relu')(z2)
        z2 = MaxPooling2D(pool_size=(2, 2), padding='same')(z2)

        decoding = Add()([decoding, z2])
        # joinedTensor = Add()([x,complement])

    decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding) # 8*8*8
    decoding = UpSampling2D((2, 2))(decoding) # 16* 16* 8

    decoding = Conv2D(16, (3, 3), activation='relu')(decoding) # 14* 14* 16
    '''
    if skip == 1 :
        z2 = Conv2D(16, (3, 3), activation='relu', padding='same')(z2)
        decoding = Add()([decoding,z2])
        #joinedTensor = Add()([x,complement])
    '''

    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoding)
    
    # Return the whole model and the encoding section as objects
    autoencoder = Model(inputs, decoding)
    encoder = Model(inputs, encoding)

    return autoencoder, encoder

def vis_data(x_train_encoded, y_train, vis_dim, n_predict, n_train, build_anim, num):
    cmap = plt.get_cmap('rainbow', 10)

    # 3-dim vis: show one view, then compile animated .gif of many angled views
    if vis_dim == 3:
        # Simple static figure
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                c=y_train[:n_predict], cmap=cmap, edgecolor='black')
        fig.colorbar(p, drawedges=True)
        #plt.show()

        # Build animation from many static figures
        if build_anim:
            angles = np.linspace(180, 360, 20)
            i = 0
            for angle in angles:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.view_init(10, angle)
                p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                        c=y_train[:n_predict], cmap=cmap, edgecolor='black')
                fig.colorbar(p, drawedges=True)
                outfile = 'anim/3dplot_step_' + chr(i + 97) + '.png'
                plt.savefig(outfile, dpi=96)
                i += 1
            call(['convert', '-delay', '50', 'anim/3dplot*', 'anim/3dplot_anim_' + str(n_train) + '.gif'])

    # 2-dim vis: plot and colorbar.
    elif vis_dim == 2:
        plt.scatter(x_train_encoded[:,0], x_train_encoded[:,1], 
                c=y_train[:n_predict], edgecolor='black', cmap=cmap)
        plt.colorbar(drawedges=True)
        #plt.show()
    if num == 1:
        plt.savefig( graph_save_path + graph_name)
    elif num == 2:
        plt.savefig(graph_save_path + graph_name+'_clustering')

if __name__ == '__main__':
    main()