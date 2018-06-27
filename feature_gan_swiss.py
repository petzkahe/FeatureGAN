# To get the current working directory into Pythonpath
# Question: What do I need it there for?

import os
import sys
import matplotlib
import numpy as np
import sklearn.datasets  # sklearn for machine learning  in python

sys.path.append(os.getcwd())
# sys.path = A list of strings that specifies the search path for modules.
# Initialized from the environment variable PYTHONPATH, plus an installation-dependent default.

# import random

matplotlib.use('Agg')  # to generate png images, alternatives: ps, pdf, svg, specify before importing pyplot
import matplotlib.pyplot as plt


# from util import argprun
#  command line overrides kwargs, see util.py

#################
# Input arguments
#################

def main():
    in_dim = 2
    dim_nn = 512  # Generator NN is then of form in_dim -> dim_nn -> dim_nn*2 -> dim_nn -> in_dim
    # Discriminator NN is then of form in_dim -> dim_nn -> dim_nn*2 -> dim_nn -> 1
    batch_size = 256
    total_no_of_iterations = 10000
    critic_no_of_iterations = 5
    penalty_weight = 11
    log_directory = "logs_"
    dataset_name = "swissroll"  # alternatives toy datasets 8gaussians, 25gaussians
    no_of_interpolation_points = 1  # the number of points btw real and fake used for the WGAN penalty

    learning_rate = 1e-4

    sampling="Gaussian"
    #For uniform sampling, define the range
    max_value = 1.0
    min_value = -1.0


    penalty_mode= "WGAN-LP"

    # Each GPU has device ID, we want to use the one numerated as 0
    # Setting environment variable CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES
    # Only necessary if i do not want to use all available computational resources
    # Must happen before importing tensorflow

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # order by id
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Print out the visible gpu devices
    from tensorflow.python.client import device_lib
    print("VISIBLE DEVICES = {}".format(str(device_lib.list_local_devices())))
    print("\n")
    print("selected GPU={}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("\n")

    import tensorflow as tf

    # External library downloaded from web
    # https://github.com/igul222/improved_wgan_training/tree/master/tflib

    import tflib as lib
    # import tflib.ops.linear
    import tflib.plot

    if os.path.exists(log_directory):
        raise Exception("The log_directory ({}) exists and should not be overwritten".format(log_directory))
    else:
        os.makedirs(log_directory)
        print("Log directory is set to {}".format(log_directory))

    # Build a shallow copy of the dictionary of local variables
    dict_local_variables = locals().copy()
    # and save in document "settings"
    with open("{}/settings".format(log_directory), "w") as writer:
        writer.write(str(dict_local_variables))

    # Later, for plotting need this part of tflib.plot
    lib.plot.logdir = log_directory

    # From tflib print some model settings out of local variables
    # via simple key matching decides what to print, can equally well just do it by hand
    # lib.print_model_settings(dict_local_variables)
    print(
        " dim = {},\n batch_size = {}, "
        "\n total_no_of_iterations =  {},"
        " \n critic_no_of_iterations =  {},"
        " \n penalty_weight =  {},"
        "  \n dataset =  {}, \n".format(dim_nn,
                                        batch_size,
                                        total_no_of_iterations,
                                        critic_no_of_iterations,
                                        penalty_weight,
                                        dataset_name))

    ################################################################
    ################################################################

    def ReLuLayer(in_dim, out_dim, input, name):

        # initalize weights

        weight_initializations = np.random.uniform(
            low=- np.sqrt(2. / in_dim),
            high=np.sqrt(2. / in_dim),
            size=(in_dim, out_dim)).astype('float32')
        bias_initializations = np.zeros(out_dim, dtype='float32')

        weights = tf.get_variable(name + ".W", initializer=weight_initializations)
        biases = tf.get_variable(name + ".b", initializer=bias_initializations)

        # To enable parameter sharing might actually need here the tflib library with lib.param()

        # assumes that input is of right size
        output = tf.matmul(input, weights)

        output = tf.nn.bias_add(output, biases)

        output = tf.nn.relu(output)

        return output

    def LinearLayer(in_dim, out_dim, input, name):

        # initalize weights

        weight_initializations = np.random.uniform(
            low=-np.sqrt(2. / in_dim),
            high=np.sqrt(2. / in_dim),
            size=(in_dim, out_dim)).astype('float32')

        bias_initializations = np.zeros(out_dim, dtype='float32')

        weights = tf.get_variable(name + ".W", initializer=weight_initializations)
        biases = tf.get_variable(name + ".b", initializer=bias_initializations)

        # To enable parameter sharing might actually need here the tflib library with lib.param

        # assumes that input is of right size
        output = tf.matmul(input, weights)
        # else:
        #    reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
        #    result = tf.matmul(reshaped_inputs, weight)
        #    result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        output = tf.nn.bias_add(output, biases)

        return output

    def Generator(noise):

        with tf.variable_scope("Generator.1", reuse=tf.AUTO_REUSE):
            output = ReLuLayer(in_dim, dim_nn, noise, "Generator.1")
        with tf.variable_scope("Generator.2", reuse=tf.AUTO_REUSE):
            output = ReLuLayer(dim_nn, 2 * dim_nn, output, "Generator.2")
        with tf.variable_scope("Generator.3", reuse=tf.AUTO_REUSE):
            output = ReLuLayer(2 * dim_nn, dim_nn, output, "Generator.3")
        with tf.variable_scope("Generator.4", reuse=tf.AUTO_REUSE):
            output = LinearLayer(dim_nn, 2, output, "Generator.4")
        return output

    def Features(inputs):
        #features = inputs
        with tf.variable_scope("Discriminator.0", reuse=tf.AUTO_REUSE):
            features = ReLuLayer(2, 64, inputs,"Discriminator.1")
        #with tf.variable_scope("Discriminator.2", reuse=tf.AUTO_REUSE):
        #    features = ReLuLayer(dim_nn, 2 * dim_nn, features, "Discriminator.2")

        return features

    def Critic(inputs):
        with tf.variable_scope("Discriminator.1", reuse=tf.AUTO_REUSE):
            output = ReLuLayer(64, dim_nn, inputs, "Discriminator.1")
        with tf.variable_scope("Discriminator.2", reuse=tf.AUTO_REUSE):
            output = ReLuLayer(dim_nn, 2 * dim_nn, output, "Discriminator.2")
        with tf.variable_scope("Discriminator.3", reuse=tf.AUTO_REUSE):
            output = ReLuLayer(2 * dim_nn, dim_nn, output, "Discriminator.3")
        with tf.variable_scope("Discriminator.4", reuse=tf.AUTO_REUSE):
            output = LinearLayer(dim_nn, 1, output, "Discriminator.4")
        return output

    # real data to be fed in later
    # shape= list [anything, in_dim] because not sure how many data points of dim in_dim I will use
    reals = tf.placeholder(tf.float32, shape=[None, 2], name='reals')

    noise = tf.placeholder(tf.float32, shape=[None, in_dim], name='noise')

    # Get fake data by passing it through Generator
    fakes = Generator(noise)

    # Compute feature representations of real and fake
    # and let Discriminator critic on all real and fake data
    features_real = Features(reals)
    critic_real = Critic(features_real)

    features_fake = Features(fakes)
    critic_fake = Critic(features_fake)

    # objectives to minimize without regularization

    objective_critic = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)

    objective_generator = -tf.reduce_mean(critic_fake)

    # WGAN gradient penalty with a distance-adaptive lambda
    #########################################
    interpolating_coefficients = np.random.uniform(
        low=0.0,
        high=1.0,
        size=[batch_size, no_of_interpolation_points]).astype('float32')

    # differences_fake_real = fakes - reals

    differences_features_fake_real = features_fake - features_real

    norm_differences_features = tf.sqrt(tf.reduce_sum(tf.square(differences_features_fake_real), axis=-1))

    # interpolating_points = np.zeros(shape=[batch_size])
    interpolating_feature_points = np.zeros(shape=[batch_size])

    penalty = 0.0

    for i in range(0, no_of_interpolation_points):
        iteration_interpolating_coefficients = interpolating_coefficients[:, i].reshape(batch_size, 1)
        interpolating_feature_points = features_real + tf.multiply(differences_features_fake_real, iteration_interpolating_coefficients)
        # interpolating_features = Features(interpolating_points)
        gradient_vectors = tf.gradients(Critic(interpolating_feature_points), interpolating_feature_points)[0]
        gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(gradient_vectors), axis=-1))

        if penalty_mode == "featureGAN":
            penalty += 1.0/no_of_interpolation_points* penalty_weight * tf.reduce_mean(
                norm_differences_features * tf.clip_by_value(gradient_norms - 1., 0., np.infty))

        # OLD WGAN-LP penalty
        elif penalty_mode == "WGAN-LP":
            penalty += 1.0 / no_of_interpolation_points * penalty_weight * tf.reduce_mean(
                tf.clip_by_value(gradient_norms - 1., 0., np.infty)**2)

        elif penalty_mode == "WGAN-GP":
            penalty += 1.0 / no_of_interpolation_points * 1.0 * tf.reduce_mean((gradient_norms - 1.) ** 2)

        else:
            raise Exception("Unknown penalty-mode: "+penalty_mode)

    all_variables = tf.trainable_variables()

    critic_and_feature_parameters = [entry for entry in all_variables if "Discriminator" in entry.name]
    generator_parameters = [entry for entry in all_variables if "Generator" in entry.name]

    train_critic = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(
        objective_critic + penalty,
        var_list=critic_and_feature_parameters
    )

    train_generator = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.5,
        beta2=0.9
    ).minimize(
        objective_generator,
        var_list=generator_parameters
    )

    for variable in dict_local_variables:
        if 'Generator' in variable:
            print ("{}: {}".format(variable.name, variable.get_shape()))
        elif "Critic" in variable:
            print ("{}: {}".format(variable.name, variable.get_shape()))

    def generate_image(samples_real, iteration_step):
        """
        Generates and saves a plot of the true distribution, the generator, and the
        critic.
        """

        if sampling == "uniform":

            n_points = 64


            if in_dim == 1:
                n_points = n_points ** 2
                points = np.zeros((n_points, 1), dtype='float32')
                points[:, 0] = np.linspace(min_value, max_value, n_points)


            elif in_dim == 2:

                #  Creates fakes for 64^2 many uniformly distributed images
                points = np.zeros((n_points, n_points, 2), dtype='float32')
                points[:, :, 0] = np.linspace(min_value, max_value, n_points)[:, None]
                # for zero: for any second entry linspace runs over first coordinate
                points[:, :, 1] = np.linspace(min_value, max_value, n_points)[None, :]
                # for one: for any first entry linspace runs over second coordinate
                points = points.reshape((-1, 2))  # gives list of points of all combinations

            elif in_dim == 3:

                n_points = 16
                points = np.zeros((n_points, n_points, n_points, 3))
                points[:, :, :, 0] = np.linspace(min_value, max_value, n_points)[:, None, None]
                points[:, :, :, 1] = np.linspace(min_value, max_value, n_points)[None, :, None]
                points[:, :, :, 2] = np.linspace(min_value, max_value, n_points)[None, None, :]
                points = points.reshape((-1, in_dim))

            else:
                print("Code for higher input dimensions not yet implemented")


        if sampling == "Gaussian":

            n_points= 32

            r = np.array(
                [0.23, 0.33, 0.4, 0.47, 0.53, 0.58, 0.63, 0.68, 0.725, 0.77, 0.815, 0.655, 0.895, 0.935, 0.98, 1.025,
                 1.07, 1.115, 1.15, 1.2, 1.245, 1.29, 1.325, 1.38, 1.43, 1.48, 1.535, 1.59, 1.645, 1.71, 1.77, 1.85,
                 1.94, 2.04, 2.13, 2.25, 2.37, 2.69, 3.5, 4])

            points = np.empty([len(r), n_points, 2])

            for j in range(len(r)):

                for i in range(n_points):
                    x = r[j] * np.cos(2 * np.pi * i / n_points)
                    y = r[j] * np.sin(2 * np.pi * i / n_points)
                    points[j, i, :] = x, y

            points = points.reshape([len(r) * n_points, 2])



        samples_fake = session.run(
            fakes,
            feed_dict={noise: points}
        )

        plt.clf()  # clear current figure

        plt.scatter(samples_real[:, 0], samples_real[:, 1], c='orange', marker='+')
        plt.scatter(samples_fake[:, 0], samples_fake[:, 1], c='green', marker='+')

        plt.savefig('{}/frame_{}.png'.format(log_directory, iteration_step))

    # generate real samples
    def generate_reals():
        if dataset_name == '25gaussians':

            dataset = []
            for i in xrange(100000 / 25):
                for x in xrange(-2, 3):
                    for y in xrange(-2, 3):
                        point = np.random.randn(2) * 0.05
                        point[0] += 2 * x
                        point[1] += 2 * y
                        dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            np.random.shuffle(dataset)
            dataset /= 2.828  # stdev
            while True:
                for i in xrange(len(dataset) / batch_size):
                    yield dataset[i * batch_size:(i + 1) * batch_size]

        elif dataset_name == 'swissroll':

            while True:
                data = sklearn.datasets.make_swiss_roll(
                    n_samples=batch_size,
                    noise=0.25
                )[0]
                data = data.astype('float32')[:, [0, 2]]
                data /= 7.5  # stdev plus a little
                yield data

        elif dataset_name == '8gaussians':

            scale = 2.
            centers = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1. / np.sqrt(2), 1. / np.sqrt(2)),
                (1. / np.sqrt(2), -1. / np.sqrt(2)),
                (-1. / np.sqrt(2), 1. / np.sqrt(2)),
                (-1. / np.sqrt(2), -1. / np.sqrt(2))
            ]
            centers = [(scale * x, scale * y) for x, y in centers]
            while True:
                dataset = []
                for i in xrange(batch_size):
                    point = np.random.randn(2) * .02
                    center = np.random.choice(centers)
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset.append(point)
                dataset = np.array(dataset, dtype='float32')
                dataset /= 1.414  # stdev
                yield dataset

    # Train loop

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        # soft placement allows cpu operations when ops not defined for gpu
        session.run(tf.initialize_all_variables())
        generate_real_samples = generate_reals()

        for iteration in xrange(total_no_of_iterations):
            # Train generator
            if iteration > 0:
                if sampling == "uniform":
                    z = np.random.uniform(low=min_value, high=max_value, size=[batch_size, in_dim]).astype('float32')
                if sampling == "Gaussian":
                    z = np.random.normal(size=[batch_size, in_dim]).astype('float32')
                _objective_generator, _ = session.run([objective_generator, train_generator], feed_dict={noise: z})

            # Train critic
            for j in xrange(critic_no_of_iterations):
                real_samples = generate_real_samples.next()
                if sampling == "uniform":
                    z = np.random.uniform(low=min_value, high=max_value, size=[batch_size, in_dim]).astype('float32')
                if sampling == "Gaussian":
                    z = np.random.normal(size=[batch_size, in_dim]).astype('float32')
                _objective_critic, _ = session.run([objective_critic, train_critic],
                                                   feed_dict={reals: real_samples, noise: z})

            # Write logs and save samples
            lib.plot.plot('critic_cost', _objective_critic)
            if iteration % 100 == 99:
                lib.plot.flush()
                generate_image(real_samples, iteration)
            lib.plot.tick()


if __name__ == "__main__":
    main()
