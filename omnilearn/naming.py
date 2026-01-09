def get_model_name(flags, fine_tune=False, add_string=""):
    model_name = "PET_{}_{}_{}_{}_{}_{}_{}{}.weights.h5".format(
        flags.dataset,
        flags.num_layers,
        "local" if flags.local else "nolocal",
        "layer_scale" if flags.layer_scale else "nolayer_scale",
        "simple" if flags.simple else "token",
        "fine_tune" if fine_tune else "baseline",
        flags.mode,
        add_string,
    )
    return model_name

