"""A basic MNIST example using Numpy and JAX.

The primary aim here is simplicity and minimal dependencies.
"""
import time

import numpy as np
import numpy.random as npr

import jax
from jax import jit, grad, random, dlpack
from jax.tree_util import tree_flatten
from jax.experimental import optimizers
import jax.numpy as jnp
import datasets
from resnet import ResNet18

import ray
import ray.util.collective as col
import cupy as cp
import os


class Dataloader:
    def __init__(self, data, target, batch_size=128):
        '''
        data: shape(width, height, channel, num)
        target: shape(num, num_classes)
        '''
        self.data = data
        self.target = target
        self.batch_size = batch_size
        num_data = self.target.shape[0]
        num_complete_batches, leftover = divmod(num_data, batch_size)
        self.num_batches = num_complete_batches + bool(leftover)

    def synth_batches(self):
        num_imgs = self.target.shape[0]
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_imgs)
            for i in range(self.num_batches):
                batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                img_batch = self.data[:, :, :, batch_idx]
                label_batch = self.target[batch_idx]
                yield img_batch, label_batch

    def __iter__(self):
        return self.synth_batches()


@ray.remote(num_gpus=1, num_cpus=1, memory=2500 * 1024 * 1024)
class Worker:
    def __init__(self):
        rng_key = random.PRNGKey(0)

        self.batch_size = 128
        self.num_classes = 10
        self.input_shape = (28, 28, 1, self.batch_size)
        self.lr = 0.01
        self.num_epochs = 10

        init_fun, predict_fun = ResNet18(self.num_classes)
        _, init_params = init_fun(rng_key, self.input_shape)

        opt_init, opt_update, get_params = optimizers.adam(self.lr)

        opt_state = opt_init(init_params)

        self.opt_state = opt_state
        self.opt_update = opt_update
        self.get_params = get_params

        self.predict_fun = predict_fun

        self.steps = 0

        # @jit
        def update(i, opt_state, batch):
            params = self.get_params(opt_state)
            gradient = grad(self.loss)(params, batch)

            ftree, tree = tree_flatten(gradient)
            for g in ftree:
                g_jdp = dlpack.to_dlpack(g)
                g_cp = cp.fromDlpack(g_jdp)
                col.allreduce(g_cp, group_name="default")
            return self.opt_update(i, grad(self.loss)(params, batch), opt_state)
        # self.update = update
        self.update = jax.jit(update)

    def init_group(self,
                   world_size,
                   rank,
                   backend="nccl",
                   group_name="default"):
        col.init_collective_group(world_size, rank, backend, group_name)

    def set_dataloader(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def run(self):
        if not self.train_dataloader:
            raise RuntimeError("Train dataloader hasn't be set.")
        if not self.test_dataloader:
            raise RuntimeError("Test dataloader hasn't be set.")

        for epoch in range(self.num_epochs):
            start_time = time.time()
            for idx, batch in enumerate(self.train_dataloader):
                self.opt_state = self.update(self.steps,
                                             self.opt_state,
                                             batch)
                self.steps+=1
            epoch_time = time.time() - start_time
            test_start_time = time.time()
            params = self.get_params(self.opt_state)
            # train_acc = self.accuracy(params, self.train_dataloader)
            test_acc = self.accuracy(params, self.test_dataloader)
            test_time = time.time() - test_start_time
            print("Epoch {} in {:0.2f} sec, test time {:0.2f} sec."
                .format(epoch, epoch_time, test_time))
            # print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))

    def loss(self, params, batch):
        inputs, targets = batch
        logits = self.predict_fun(params, inputs)
        return -jnp.sum(logits * targets)

    def accuracy(self, params, dataloader):
        result = []
        for _, (inputs, targets) in enumerate(dataloader):
            logits = self.predict_fun(params, inputs)
            predicted_class = jnp.argmax(logits, axis=1)
            target_class = jnp.argmax(targets, axis=1)
            result.append(jnp.mean(predicted_class == target_class))
        return np.array(result).mean()


if __name__ == "__main__":
    gpu_ids = [0,1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    num_gpus = len(gpu_ids)

    ray.init(num_gpus=num_gpus, num_cpus=4, local_mode=True)

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28).transpose(2, 3, 1, 0)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28).transpose(2, 3, 1, 0)

    train_dataloader = Dataloader(train_images, train_labels, batch_size=32)
    test_dataloader = Dataloader(test_images, test_labels, batch_size=32)

    actors = [Worker.remote() for _ in range(num_gpus)]

    ray.get([actor.init_group.remote(num_gpus, rank, group_name="default")
             for rank, actor in enumerate(actors)])

    ray.get([actor.set_dataloader.remote(train_dataloader, test_dataloader)
            for actor in actors])

    ray.get([actor.run.remote() for actor in actors])

