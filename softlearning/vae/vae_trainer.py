from os import path as osp
import numpy as np
import tensorflow as tf
import math
from numbers import Number
from collections import OrderedDict


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


"""
Implementation of save_image() and make_grid() of PyTorch in numpy

Original Pytorch implementation can be found here:
https://github.com/pytorch/vision/blob/master/torchvision/utils.py
"""
def make_grid(img, nrow=8, padding=2, normalize=False, irange=None, scale_each=False, pad_value=0):

    if isinstance(img, list):
        img = np.concatenate(img, axis=0)

    if img.ndim == 2:
        img = np.expand_dims(img, 0)

    if img.ndim == 3:
        if img.shape[2] == 1:
            img = np.concatenate((img, img, img), axis=2)
        img = np.expand_dims(img, 0)

    if img.ndim == 4 and img.shape[3] == 1:
        img = np.concatenate((img, img, img), axis=3)

    if normalize is True:
        img = np.array(img, copy=True)

        def norm_ip(img, min, max):
            np.clip(img, min, max, out=img)
            np.add(img, -min, out=img)
            np.divide(img, max - min + 1e-5, out=img)

        def norm_range(t, irange):
            if irange is not None:
                norm_ip(t, irange[0], irange[1])
            else:
                norm_ip(t, np.min(t), np.max(t))

        if scale_each is True:
            for x in img:
                norm_range(x, irange)
        else:
            norm_range(x, irange)

    if img.shape[0] == 1:
        return np.squeeze(img, axis=0)
                
    nmaps = img.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(img.shape[1] + padding), int(img.shape[2] + padding)
    grid = np.ones((height * ymaps + padding, width * xmaps + padding, 3)) * pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height + padding : (y+1) * height,\
                 x * width + padding : (x+1) * width,\
                 :] = img[k]
            k = k + 1
    return grid

def save_image(img, filename, nrow=8, padding=2,
               normalize=False, irange=None, scale_each=False, pad_value=0):
    from PIL import Image
    grid = make_grid(img, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, irange=irange, scale_each=scale_each)

    np.multiply(grid, 255, out=grid)
    np.add(grid, 255, out=grid)
    np.clip(grid, 0, 255, out=grid)
    grid = grid.astype(np.uint8)
    im = Image.fromarray(grid)
    im.save(filename)

def normalize_image(image):
    """
    rlkit imports this method from multiworld
    Implementation taken from https://github.com/vitchyr/multiworld/blob/master/multiworld/core/image_env.py
    """
    return np.float32(image) / 255.0

class ConvVAETrainer():
    def __init__(
            self,
            model,
            data_keys,
            train_dataset=np.array([], dtype=np.uint8),
            test_dataset=np.array([], dtype=np.uint8),
            batch_size=128,
            log_interval=0,
            beta=0.5,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            session=None
    ):

        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot
        self.data_keys = data_keys

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength
        self._session = session or tf.keras.backend.get_session()

        self.lr = lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, epsilon=1e-8)

        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset.dtype == np.uint8
        assert self.test_dataset.dtype == np.uint8

        self.batch_size = batch_size

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )

        self.vae_input_ph = tf.placeholder(tf.float32, [None, model.imlength])
        self.rand_input_test = tf.placeholder(tf.float32, [None, self.representation_size])

        self.random_decoded, _ = self.model.decode(self.rand_input_test, training=False)

        self.encoder_mu_notrain, self.encoder_var_notrain = self.model.encode(self.vae_input_ph, training=False)

        self.recons_train, self.obs_params_train, self.latent_params_train = self.model(self.vae_input_ph)
        self.log_prob_train = self.model.logprob(self.vae_input_ph, self.obs_params_train)
        self.kle_train = self.model.kl_divergence(self.latent_params_train)
        self.loss_train = self.beta * self.kle_train - self.log_prob_train
        self.update_op = self.optimizer.minimize(self.loss_train, var_list=self.model.trainable_variables)

        self.recons_test, self.obs_params_test, self.latent_params_test = self.model(self.vae_input_ph, training=False)
        self.log_prob_test = self.model.logprob(self.vae_input_ph, self.obs_params_test, training=False)
        self.kle_test = self.model.kl_divergence(self.latent_params_test, training=False)
        self.loss_test = self.beta * self.kle_test - self.log_prob_test

    def get_dataset_stats(self, data):
        data = normalize_image(data)
        mus = self._session.run(self.encoder_mu, feed_dict={self.vae_input_ph : data})
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def _kl_np_to_np(self, np_imgs):
        data = normalize_image(np_imgs)
        mu, log_var = self._session.run([self.encoder_mu, self.encoder_var], feed_dict={self.vae_input_ph : data})
        return -np.sum(1 + log_var - np.power(mu, 2) - np.exp(log_var), axis=1)

    def _reconstruction_squared_error_np_to_np(self, np_imgs):
        data = normalize_image(np_imgs)
        recons = self._session.run(self.recons_train, feed_dict={self.vae_input_ph : data})
        error = data - recons
        return np.sum(error ** 2, axis=1)

    def set_vae(self, vae):
        """
        This won't work with TF
        This function isn't used in rlkit so leaving it unchanged

        TODO: Delete all current ops
              Use ops from the new VAE
        """
        self.model = vae

    def get_batch(self, training=True):
        dataset = self.train_dataset if training else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :])
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            samples = samples - self.train_data_mean
        return samples

    def get_debug_batch(self, training=True):
        """
        This is most likely a bug in the original code
        Not sure about what the
        `X, Y = dataset` does

        This method is never called in the original code so it should not be a problem
        """
        dataset = self.train_dataset if training else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return X, Y


    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        training = True
        losses = []
        log_probs = []
        kles = []
        for batch_idx in range(batches):
            if sample_batch is not None:
                data = sample_batch(self.batch_size)
                next_obs = np.concatenate([
                    data[key] for key in self.data_keys], axis=0)
            else:
                next_obs = self.get_batch()

            _, loss, log_prob, kle = self._session.run([self.update_op, self.loss_train, self.log_prob_train, self.kle_train],
                                                feed_dict={self.vae_input_ph : next_obs})

            #print(loss)
            
            losses.append(loss)
            log_probs.append(log_prob)
            kles.append(kle)

            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                                      epoch,
                                      loss / len(next_obs)))

        if from_rl:
            self.vae_logger_stats_for_rl['Train VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Train VAE Log Prob'] = np.mean(log_probs)
            self.vae_logger_stats_for_rl['Train VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Train VAE Loss'] = np.mean(losses)
        else:
            """Based on logger in rlkit core so skipped for now"""
            #  logger.record_tabular("train/epoch", epoch)
            #  logger.record_tabular("train/Log Prob", np.mean(log_probs))
            #  logger.record_tabular("train/KL", np.mean(kles))
            #  logger.record_tabular("train/loss", np.mean(losses))


    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_vae=True,
            from_rl=False,
            save_dir='.'
    ):
        training = False
        losses = []
        log_probs = []
        kles = []
        zs = []
        for batch_idx in range(10):
            next_obs = self.get_batch(training=training)

            loss, log_prob, kle, latent_distribution_params, reconstructions = self._session.run([self.loss_test,
                                                                                         self.log_prob_test,
                                                                                         self.kle_test,
                                                                                         self.latent_params_test,
                                                                                         self.recons_test],
                                                                                        feed_dict={self.vae_input_ph : next_obs})

            encoder_mean = latent_distribution_params[0]
            z_data = encoder_mean
            for i in range(len(z_data)):
                zs.append(z_data[i, :])

            losses.append(loss)
            log_probs.append(log_prob)
            kles.append(kle)

            if batch_idx == 0 and save_reconstruction:
                n = min(next_obs.shape[0], 8)

                comparison = np.concatenate(
                                np.reshape(next_obs[:n, :self.imlength], [-1, self.imsize, self.imsize, self.input_channels]),
                                np.reshape(reconstructions, [self.batch_size, self.imsize, self.imsize, self.input_channels])[:n]
                             )
                    
                """Based on logger in rlkit core so skipped for now"""
                #  save_dir = osp.join(logger.get_snapshot_dir(),
                                    #  'r%d.png' % epoch)

                save_dir = osp.join(save_dir, 'r%d.png' % epoch)
                # TODO: save_image implementation check
                save_image(comparison.numpy(), save_dir, nrow=n)

        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)

        if from_rl:
            self.vae_logger_stats_for_rl['Test VAE Epoch'] = epoch
            self.vae_logger_stats_for_rl['Test VAE Log Prob'] = np.mean(
                log_probs)
            self.vae_logger_stats_for_rl['Test VAE KL'] = np.mean(kles)
            self.vae_logger_stats_for_rl['Test VAE loss'] = np.mean(losses)
            self.vae_logger_stats_for_rl['VAE Beta'] = self.beta
        else:
            """Based on logger in rlkit core so skipped for now"""
            #  for key, value in self.debug_statistics().items():
                #  logger.record_tabular(key, value)
#
            #  logger.record_tabular("test/Log Prob", np.mean(log_probs))
            #  logger.record_tabular("test/KL", np.mean(kles))
            #  logger.record_tabular("test/loss", np.mean(losses))
            #  logger.record_tabular("beta", self.beta)
#
            #  logger.dump_tabular()
            #  if save_vae:
                #  logger.save_itr_params(epoch, self.model)

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn’t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        training = False
        debug_batch_size = 64
        data = self.get_batch(training=training)
        reconstructions, _, _ = self.model(data, training=training)
        reconstructions = self._session.run(self.recons_test, feed_dict={self.vae_input_ph : data})
        img = data[0]

        recon_mse = np.mean((reconstructions[0] - img) ** 2)

        # TODO: This is a work around for torch.expand in the original code, find if any better one exists
        img_repeated = np.repeat(img, debug_batch_size)
        img_repeated = np.reshape(img_repeated, (debug_batch_size, -1))

        samples = np.random.randn(debug_batch_size, self.representation_size).astype(np.float32)
        random_imgs = self._session.run(self.random_decoded, feed_dict={self.rand_input_test : samples})
        random_mses = np.mean((random_imgs - img_repeated) ** 2)
        mse_improvement = np.mean(random_mses, axis=1) - recon_mse

        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            random_mses,
        ))
        stats['debug/MSE of reconstruction'] = recon_mse
        return stats

    def dump_samples(self, epoch, save_dir='.'):
        training = False
        sample = np.random.randn(64, self.representation_size).astype(np.float32)
        sample = self._session.run(self.random_decoded, feed_dict={self.rand_input_test : sample})
        """Based on logger in rlkit core so skipped for now"""
        #  save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_dir = osp.join(save_dir, 's%d.png' % epoch)
        save_image(
                np.reshape(sample, (64, self.imsize, self.imsize, self.input_channels)),
            save_dir
        )

    def _dump_imgs_and_reconstructions(self, idxs, save_dir, filename):
        training = False
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.train_dataset[i]
            img_np = normalize_image(img_np)
            recon = self._session.run(self.recons_test, feed_dict={self.vae_input_ph : img_np})

            img = np.reshape(img_np, (self.imsize, self.imsize, self.input_channels))
            rimg = np.reshape(recon, (self.imsize, self.imsize, self.input_channels))
            imgs.append(img)
            recons.append(rimg)

        all_imgs = np.stack(imgs + recons)

        """Based on logger in rlkit core so skipped for now"""
        #  save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_file = osp.join(save_dir, filename)
        save_image(
            all_imgs.numpy(),
            save_file,
            nrow=4,
        )
