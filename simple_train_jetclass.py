import logging
import os
import pickle

from omnilearn.distributed import setup_gpus
from omnilearn.naming import get_model_name
from dataclasses import dataclass
from omnilearn.data.loaders import JetClassDataLoader
from omnilearn.models.pet import PET
from tensorflow import keras

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class params:
    dataset = "jetclass"
    data_dir = "/data/raw_jetclass"
    model_save_dir = "/data/models/omnilearn_test"
    mode = "all"
    batch_size = 250
    epoch = 200
    warm_epoch = 3
    stop_epoch = 30
    lr = 3e-5
    wd = 1e-5
    b1 = 0.95
    b2 = 0.99
    lr_factor = 10.0
    nid = 0
    fine_tune = False
    local = False
    num_layers = 2  # 8
    drop_probability = 0.0
    simple = False
    talking_head = False
    layer_scale = False


default_params = params()


def configure_optimizers(
    default_params: params,
    train_loader: JetClassDataLoader,
    lr_factor=1.0,
):
    nevts_by_batch = train_loader.nevts // default_params.batch_size

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=default_params.lr / lr_factor,
        warmup_steps=default_params.warm_epoch * nevts_by_batch,
        decay_steps=default_params.epoch * nevts_by_batch,
    )

    optimizer = keras.optimizers.Lion(
        learning_rate=lr_schedule,
        weight_decay=default_params.wd * lr_factor,
        beta_1=default_params.b1,
        beta_2=default_params.b2,
    )

    return optimizer


def main():
    setup_gpus()

    # For now only train on the validation set to speed up testing
    train_loader = JetClassDataLoader(
        os.path.join(default_params.data_dir, "val"),
        default_params.batch_size,
    )

    model = PET(
        num_feat=train_loader.num_feat,
        num_jet=train_loader.num_jet,
        num_classes=train_loader.num_classes,
        local=default_params.local,
        num_layers=default_params.num_layers,
        drop_probability=default_params.drop_probability,
        simple=default_params.simple,
        layer_scale=default_params.layer_scale,
        talking_head=default_params.talking_head,
        mode=default_params.mode,
    )

    if default_params.fine_tune:
        model_name = (
            get_model_name(default_params, default_params.fine_tune)
            .replace(default_params.dataset, "jetclass")
            .replace("fine_tune", "baseline")
            .replace(default_params.mode, "all")
        )

        model_path = os.path.join(
            default_params.model_save_dir,
            "checkpoints",
            model_name,
        )
        logger.info(f"Loading model weights from {model_path}")
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    optimizer_head = configure_optimizers(
        default_params,
        train_loader,
    )

    optimizer_body = configure_optimizers(
        default_params,
        train_loader,
        lr_factor=default_params.lr_factor if default_params.fine_tune else 1.0,
    )

    model.compile(optimizer_body, optimizer_head)

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=default_params.stop_epoch,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=200,
            min_lr=1e-6,
        ),
    ]

    checkpoint_name = get_model_name(
        default_params,
        default_params.fine_tune,
        add_string=("_{}".format(default_params.nid) if default_params.nid > 0 else ""),
    )
    checkpoint_path = os.path.join(
        default_params.model_save_dir, "checkpoints", checkpoint_name
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        mode="auto",
        save_weights_only=True,
        period=1,
    )
    callbacks.append(checkpoint_callback)

    training_history = model.fit(
        train_loader.make_tfdata(),
        epochs=default_params.epoch,
        validation_data=train_loader.make_tfdata(),
        batch_size=default_params.batch_size,
        callbacks=callbacks,
        steps_per_epoch=train_loader.steps_per_epoch,
        validation_steps=train_loader.steps_per_epoch,
        verbose=True,
    )

    history_save_dir = os.path.join(default_params.model_save_dir, "histories")
    history_name = get_model_name(
        default_params,
        default_params.fine_tune,
    ).replace(".weights.h5", ".pkl")

    if not os.path.exists(history_save_dir):
        os.makedirs(history_save_dir)

    with open(os.path.join(history_save_dir, history_name), "wb") as f:
        pickle.dump(training_history.history, f)


if __name__ == "__main__":
    main()
