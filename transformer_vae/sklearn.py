from collections import namedtuple
from sklearn import svm
import numpy as np


Dataset = namedtuple("Dataset", ["latent", "class_label"])


def un_batch(latents_with_class):
    flat_latents, flat_classes = [], []
    for latents, classes in latents_with_class:
        flat_latents += latents
        flat_classes += classes
    return Dataset(np.array(flat_latents), np.array(flat_classes))


def train_test_split(dataset: list, test_ratio):
    assert dataset.latent.shape[0] == dataset.class_label.shape[0]
    split = int(dataset.latent.shape[0] * test_ratio)
    return [
        Dataset(dataset.latent[split:], dataset.class_label[split:]),
        Dataset(dataset.latent[:split], dataset.class_label[:split]),
    ]


def train_svm(latents_with_class):
    latents_with_class = un_batch(latents_with_class)
    train, test = train_test_split(latents_with_class, 0.3)
    clf = svm.SVC()
    clf.fit(train.latent, train.class_label)
    predicted_classes = clf.predict(test.latent)
    random_predictions = np.random.randint(
        train.class_label.min(), train.class_label.max(), size=predicted_classes.shape
    )
    return {
        "svm_classification_test_accuracy": (predicted_classes == test.class_label).sum() / test.class_label.shape[0],
        "random_classification_test_accuracy": (random_predictions == test.class_label).sum()
        / test.class_label.shape[0],
    }


def t_sne(self, latents_with_class):
    # TODO get t-sne plot then return points for W&B
    # for i, perplexity in enumerate([5, 30, 50, 100]):
    pass
