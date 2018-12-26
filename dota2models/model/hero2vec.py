import ignite.engine as igne
import ignite.metrics as ignm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


MAX_HERO_ID = 121


class Hero2Vec(nn.Module):
    # we input 4 heroes and the network should output the fifth one
    INPUT_HEROES = 4

    def __init__(self, emb_dim: int = 128, hero_count: int = MAX_HERO_ID + 1):
        super().__init__()

        self.emb_dim = emb_dim
        self.hero_count = hero_count

        self.hero_embedding = nn.Linear(hero_count, emb_dim, bias=False)
        self.missing_hero_softmax = nn.Linear(emb_dim, hero_count)

    def forward(self, inputs):
        embedded_heroes = self.hero_embedding(inputs)
        team_embedding = torch.mean(embedded_heroes, dim=1)
        missing_hero = self.missing_hero_softmax(team_embedding)
        return missing_hero


class HeroDataset:
    def __init__(self):
        self.inputs = np.load('data/processed/inputs.npy').astype('float32')
        self.labels = np.load('data/processed/labels.npy').astype('long')

        assert len(self.inputs) == len(self.labels)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __len__(self):
        return len(self.inputs)


def get_train_loader(batch_size=256):
    dataset = HeroDataset()

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    return train_loader


def train():
    model = Hero2Vec()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.99)
    loss = nn.CrossEntropyLoss()
    trainer = igne.create_supervised_trainer(model, optimizer, loss)
    evaluator = igne.create_supervised_evaluator(
        model,
        metrics={
            'accuracy': ignm.CategoricalAccuracy(),
            'loss': ignm.Loss(loss),
        },
    )

    train_loader = get_train_loader(batch_size=1024)

    @trainer.on(igne.Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        if trainer.state.iteration % 100 == 0:
            print(f'[Epoch {trainer.state.epoch}] Loss: {trainer.state.output:.2f}')

    @trainer.on(igne.Events.EPOCH_COMPLETED)
    def log_training_evaluation(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f'[Epoch {trainer.state.epoch}] Accuracy: {metrics["accuracy"]}, Loss: {metrics["loss"]}')

    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f'[Before training] Accuracy: {metrics["accuracy"]}, Loss: {metrics["loss"]}')

    trainer.run(train_loader, max_epochs=10)

    import pudb; pu.db
    pass


if __name__ == '__main__':
    train()
