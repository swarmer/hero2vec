import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

tf.enable_eager_execution()


MAX_HERO_ID = 121


class Hero2Vec(tfk.Model):
    # we input 4 heroes and the network should output the fifth one
    INPUT_HEROES = 4

    def __init__(self, emb_dim: int = 128, hero_count: int = MAX_HERO_ID + 1):
        super().__init__(name='hero2vec')

        self.emb_dim = emb_dim
        self.hero_count = hero_count

        self.embedding_weights = self.add_weight('emb_weights', (hero_count, emb_dim), 'float32')
        self.missing_hero_logits = tfk.layers.Dense(hero_count, activation='softmax')

    def call(self, inputs):
        print(inputs)
        embedded_heroes = self.embedding_weights @ inputs
        team_embedding = tf.reduce_mean(embedded_heroes, axis=1)
        missing_hero = self.missing_hero_logits(team_embedding)
        return missing_hero

    # def compute_output_shape(self, input_shape):
    #     input_shape = tf.TensorShape(input_shape).as_list()
    #     shape = (input_shape[0], self.hero_count)
    #     return tf.TensorShape(shape)


class HeroDataset(tfk.utils.Sequence):
    def __init__(self):
        super().__init__()
        self.inputs = np.load('data/processed/inputs.npy').astype('float32')
        self.labels = np.load('data/processed/labels.npy').astype('long')

        assert len(self.inputs) == len(self.labels)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)


def train():
    model = Hero2Vec()
    model.compile(
        tf.train.MomentumOptimizer(0.1, 0.99, use_nesterov=True),
        loss='sparse_categorical_crossentropy',
    )

    dataset = HeroDataset()

    gen = list(zip(dataset.inputs, dataset.labels))

    model.fit_generator(
        gen,
        epochs=10,
        #batch_size=1024,
    )

    import pudb; pu.db
    pass


if __name__ == '__main__':
    train()
