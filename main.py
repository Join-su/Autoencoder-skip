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
from keras.utils import np_utils
from keras.layers import Lambda, Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, \
    Reshape, Flatten, Add, concatenate
from keras import optimizers, Sequential
from keras.models import load_model
from utils import *

from keras.datasets import mnist
import os
from PIL import Image

labels_val = ['u4e00', 'u4e4b', 'u4eba', 'u4ee5', 'u53ef', 'u540c', 'u5b50', 'u5df2', 'u6240', 'u660e', 'u672a', 'u7121', 'u751f', 'u81ea', 'u898b', 'u9593']

BatchSize = 32
Epochs = 25

skip = 1
deno = 0
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
else:
    en_save = 'nom_{}_encoder'.format(Epochs)
    auto_save = 'nom_{}_autoencoder'.format(Epochs)
    graph_name = 'nom_{}_graph'.format(Epochs)
    img_name = 'nom_{}_img'.format(Epochs)


def main():
    global labels_val
    # Parameters: whether to train a new neural network (or load already-trained from disk),
    # how many to train, how many to predict (or load from disk) and visualize (dimension? whether to build 3d .gif?)
    train_new = False
    n_train = 24000
    predict_new = True
    n_predict = 3000
    vis_dim = 2
    build_anim = False

    # Load MNIST dataset.
    # x_train, y_train, x_test, y_test = import_format_data()
    #
    # x_train = x_train.astype(np.float)
    # x_train = x_train.reshape(60000, 28, 28, 1)
    # x_test = x_test.astype(np.float)
    # x_test = x_test.reshape(10000, 28, 28, 1)

    #Img_Path = "C:\\Users\\ialab\\Desktop\\hanja_data\\"
    Img_Path = "C:\\Users\\ialab\\Desktop\\hanja_data_deno\\"
    if deno == 1 :
        save_img_path = 'C:\\Users\\ialab\\Desktop\\hanja_data_deno\\'
    elif skip == 1:
        save_img_path = 'C:\\Users\\ialab\\Desktop\\hanja_data_skip\\'




    x_train, y_train, result = data_set_fun(Img_Path, 0)
    print('y_train :', y_train)
    # print('train_분포 : ', result)
    x_test, y_test, result = data_set_fun(Img_Path, 0)
    # print('test_분포 : ', result)

    #labels_val = list(set(labels_val))
    #labels_val.sort()
    #print(labels_val)
    labels_count = len(labels_val)

    #y_train = index_label(y_train)
    #y_test = index_label(y_test)

    #print('y_train :',y_train)
    '''
    #print(len(teY), len(trY))
    trY = dence_to_one_hot(trY, labels_count)
    teY = dence_to_one_hot(teY, labels_count)
    # number of classes
    '''
    #labels_count += 1
    #y_train = np_utils.to_categorical(y_train, labels_count)
    #y_test = np_utils.to_categorical(y_test, labels_count)

    print('y_train :', y_train)
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
        # autoencoder, encoder = build_autoencoder((x_train.shape[1],), encoding_dim=30)
        autoencoder, encoder = build_autoencoder((x_train.shape[1],), encoding_dim=30)
        autoencoder.compile(optimizer=optimizers.Adadelta(), loss='mean_squared_error')

        autoencoder.summary()
        autoencoder.fit(x_train[:n_train], x_train[:n_train], epochs=Epochs, batch_size=BatchSize)
        autoencoder.save(save_path + auto_save)
        encoder.save(save_path + en_save)
    else:
        encoder = load_model(save_path + en_save)
        autoencoder = load_model(save_path + auto_save, custom_objects={'tf': tf})



        autoencoder_last_layer = autoencoder.layers[-1].get_weights()
        #conv_out = autoencoder.layers[-5].output ## skip part Normal_branch
        #conv_out = autoencoder.layers[-4].output  ## skip part skip_branch

        conv_out = autoencoder.layers[-3].output ## denoising part conv2d_6

        # conv2_8_output = Add(name='test_add_1')([conv2_7_output,conv2_8_output])

        x = UpSampling2D((2, 2), name='test_UpSampling2D_1')(conv_out)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='test_conv2d_1')(x)
        test_model = Model(autoencoder.inputs, x)
        test_model.compile(optimizer=optimizers.Adadelta(), loss='mean_squared_error')
        test_model.summary()
        print(np.shape(autoencoder_last_layer))
        # autoencoder_last_layer = Lambda(lambda x: tf.reshape(x, shape=[3,3,16,1]))(autoencoder_last_layer)
        # test_model.set_weights(autoencoder_last_layer)


    decoded_imgs = autoencoder.predict(x_test)
    #decoded_imgs = test_model.predict(x_test)

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
    '''
    if predict_new:
        x_train_predict = encoder.predict(x_test[:n_predict])
        # x_train_predict = x_test[:n_predict]
        # x_train_predict = np.reshape(x_train_predict,(n_predict,784))
        print(np.shape(x_train_predict))

        print("Performing t-SNE dimensionality reduction...")
        x_train_encoded = TSNE(n_components=vis_dim).fit_transform(x_train_predict)
        np.save(save_path + '%sx_%sdim_tnse_%s.npy' % (n_predict, vis_dim, n_train), x_train_encoded)
        print("Done.")
     else:
         x_train_encoded = np.load(str(n_predict) + 'x_' + str(vis_dim) + 'dim_tnse_' + str(n_train) + '.npy')

    vis_data(x_train_encoded, y_test, vis_dim, n_predict, n_train, build_anim, 1)  # 정답
    
    

    if vis_dim == 2:
        model = KMeans(init="k-means++", n_clusters=16, random_state=0)
        model.fit(x_train_encoded)
        y_pred = model.labels_

        # count = 0
        #
        # for j in range(len(y_pred)) :
        #     if y_pred[j] == y_test[j] :
        #         count += 1
        #
        # print('정확도 : ', (count/len(y_pred)))

        vis_data(x_train_encoded, y_pred, vis_dim, n_predict, n_train, build_anim, 2)  # Clusturing
    '''
    '''
    if test == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/test_{}.png'.format("skip_connections", Epochs))
    elif skip == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/branch_norm_{}.png'.format("skip_connections", Epochs))
        # save_images(decoded_imgs[:100], image_manifold_size(100), './{}/skip_test_{}.png'.format("skip_connections", Epochs))
        print('save file')
    elif deno == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/part_deno_{}.png'.format("skip_connections", Epochs))
    else:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/nom_{}.png'.format("skip_connections", Epochs))

        # x_test = decoded_imgs
    '''

    #print(np.shape(decoded_imgs))
    for i,img1 in enumerate(decoded_imgs) :
        import scipy.misc
        import cv2
        img1 = np.array(img1)
        #ret, img1 = cv2.threshold((img1*255), 127, 255, cv2.THRESH_BINARY)
        color_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        #ret, color_img = cv2.threshold((color_img * 255), 127, 255, cv2.THRESH_BINARY)
        rgb = scipy.misc.toimage(color_img)
        #plt.imshow(img1, 'gray')
        #plt.savefig(save_img_path + y_test[i])
        scipy.misc.imsave(save_img_path + y_test[i], rgb)


        # print(np.shape(img1))
        # print(type(img1))
        # #print(img1)
        # image = Image.fromarray((img1 * 255).astype(np.uint8), 'Gray')
        # # img1_size = np.shape(img1)
        # # img = Image.new('RGB', (img1_size[0],img1_size[1]), "white")
        # # img.putdata(img1)
        # # #img.show()
        # image.save(save_img_path + 'img_%d.png' %i)



def dataset(images):
    # data = pd.read_csv(PATH, header=None)
    # images = data.iloc[:, :].values
    images = images.astype(np.float)
    images = images.reshape(100, 100, 1)
    #images = np.multiply(images, 1.0 / 255.0)

    return images

def data_set_fun(path, set_size):

    train = True
    filename_list = os.listdir(path)
    if set_size == 0:
        set_size = len(filename_list)
        train = False

    X_set = np.empty((set_size, 100, 100, 1), dtype=np.float32)
    Y_set = np.empty((set_size), dtype=np.float32)
    name = []

    np.random.shuffle(filename_list)
    result = dict()

    for i, filename in enumerate(filename_list):
        if i >= set_size:
            break
        # name.append(filename)
        label = filename.split('.')[0]
        #print(label)
        # label = label.split('_')[2]
        label = label.split('_')[-1]
        #print(label)
        result[label] = result.setdefault(label, 0) + 1
        # print("label",label)
        name.append(filename)
        #Y_set[i] = label
        #Y_set[i] = filename

        file_path = os.path.join(path, filename)
        img = Image.open(file_path)
        img = img.convert('1')  # convert image to black and white
        imgarray = np.array(img)
        imgarray = imgarray.flatten()
        # print(imgarray)

        images = dataset(imgarray)

        X_set[i] = images

        #labels_val.append(label)

    # if train:
    #    return X_set, Y_set, result
    #Y_set = index_label(name)
    return X_set, name, result

def dence_to_one_hot(labels_dence, num_classes):
    # print(labels_dence)
    num_labes = labels_dence.shape[0]
    # print(num_labes)
    index_offset = np.arange(num_labes) * num_classes
    # print(index_offset)
    labels_one_hot = np.zeros((num_labes, num_classes))
    # print(labels_dence.ravel())
    labels_one_hot.flat[index_offset + labels_dence.ravel()] = 1  # flat - 배열을 1차원으로 두고 인덱스를 이용해 값 확인
    return labels_one_hot

def index_label(label):
    # print(label)
    list = []
    for j in range(len(label)):
        for i in range(len(labels_val)):
            if label[j] == labels_val[i]:
                list.append(i)
                break
    return np.asarray(list)


def import_format_data():
    # Get dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Turn [0,255] values in (N, A, B) array into [0,1] values in (N, A*B) flattened arrays
    x_train = x_train.astype('float64') / 255.0
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.astype('float64') / 255.0
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

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

        # return x_train_noisy, y_train, x_test_noisy, y_test
        return x_train_noisy, y_train, x_test, y_test

    return x_train, y_train, x_test, y_test


def build_autoencoder(input_shape, encoding_dim):
    # Activation function: selu for SNN's: https://arxiv.org/pdf/1706.02515.pdf
    encoding_activation = 'selu'
    decoding_activation = 'selu'

    # Preliminary parameters
    # inputs = Input(shape=input_shape) #Tensor("input_1:0", shape=(?, 784), dtype=float32)
    # feat_dim = input_shape[0] #784

    inputs = Input(shape=(100, 100, 1,))

    # Encoding layers: successive smaller layers, then a batch normalization layer.
    # Conv1 #
    encoding = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    z2 = encoding

    # Conv2 #
    encoding = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    #z2 = encoding

    # Conv3 #
    encoding = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    z1 = encoding
    # print(encoding)
    # Reshape((sequence_length, embedding_dimension, 1), input_shape = (sequence_length, embedding_dimension))
    flatten_layer = Flatten()  # instantiate the layer
    encoding = flatten_layer(encoding)  # call it on the given tensor
    feat_dim = 13 * 13 * 8

    encoding = Dense(feat_dim, activation=encoding_activation, kernel_initializer='lecun_normal')(encoding)
    encoding = Dense(int(feat_dim / 2), activation=encoding_activation)(encoding)
    # encoding = Dense(int(feat_dim/4), activation=encoding_activation)(encoding)
    encoding = Dense(encoding_dim, activation=encoding_activation)(encoding)

    # Decoding layers for reconstruction
    # decoding = Dense(int(feat_dim/4), activation=decoding_activation)(encoding)
    decoding = Dense(int(feat_dim / 2), activation=decoding_activation)(encoding)
    decoding = Dense(feat_dim, activation=decoding_activation)(decoding)
    print(decoding)

    decoding = Lambda(lambda x: tf.reshape(x, shape=[-1, 13, 13, 8]))(decoding)
    # decoding = Reshape((-1,7,7,8),input_shape = (BatchSize,7*7*8))(decoding)  # (,392)
    print('1 : ', decoding)


    if skip == 1:
        z1 = Conv2D(8, (3, 3), activation='relu', padding='same')(z1)
        # z1 = Conv2D(8, (1, 1), activation='relu', padding='same')(z1)
        decoding = Add()([decoding, z1])
        # joinedTensor = Add()([x,complement])


    print('2 :', decoding)

    decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)  # 4*4*8
    decoding = UpSampling2D((2, 2))(decoding)  # 8*8*8

    '''
    if skip == 1:
        z2 = UpSampling2D((2, 2))(z2)
        z2 = Conv2DTranspose(8, (3, 3), activation='relu')(z2)
        z2 = MaxPooling2D(pool_size=(2, 2), padding='same')(z2)

        decoding = Add()([decoding, z2])
        # joinedTensor = Add()([x,complement])
    '''


    decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)  # 8*8*8
    decoding = UpSampling2D((2, 2))(decoding)  # 16* 16* 8

    decoding = Conv2D(16, (3, 3), activation='relu')(decoding)  # 14* 14* 16


    if skip == 1 :
        z2 = Conv2D(16, (3, 3), activation='relu', padding='same')(z2)
        decoding = Add()([decoding,z2])
        #joinedTensor = Add()([x,complement])



    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoding)

    # Return the whole model and the encoding section as objects
    autoencoder = Model(inputs, decoding)
    encoder = Model(inputs, encoding)

    return autoencoder, encoder


def vis_data(x_train_encoded, y_train, vis_dim, n_predict, n_train, build_anim, num):
    cmap = plt.get_cmap('rainbow', 16)

    # 3-dim vis: show one view, then compile animated .gif of many angled views
    if vis_dim == 3:
        # Simple static figure
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(x_train_encoded[:, 0], x_train_encoded[:, 1], x_train_encoded[:, 2],
                         c=y_train[:n_predict], cmap=cmap, edgecolor='black')
        fig.colorbar(p, drawedges=True)
        # plt.show()

        # Build animation from many static figures
        if build_anim:
            angles = np.linspace(180, 360, 20)
            i = 0
            for angle in angles:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.view_init(10, angle)
                p = ax.scatter3D(x_train_encoded[:, 0], x_train_encoded[:, 1], x_train_encoded[:, 2],
                                 c=y_train[:n_predict], cmap=cmap, edgecolor='black')
                fig.colorbar(p, drawedges=True)
                outfile = 'anim/3dplot_step_' + chr(i + 97) + '.png'
                plt.savefig(outfile, dpi=96)
                i += 1
            call(['convert', '-delay', '50', 'anim/3dplot*', 'anim/3dplot_anim_' + str(n_train) + '.gif'])

    # 2-dim vis: plot and colorbar.
    elif vis_dim == 2:
        plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1],
                    c=y_train[:n_predict], edgecolor='black', cmap=cmap)
        plt.colorbar(drawedges=True)
        # plt.show()
    if num == 1:
        plt.savefig(graph_save_path + graph_name)
    elif num == 2:
        plt.savefig(graph_save_path + graph_name + '_clustering')


if __name__ == '__main__':
    main()
