"""
Model utilities (Paddle Version)
"""
from src.backbones.utae import UTAE
from src.backbones.utae import RecUNet

"""
Get the model based on configuration
"""


def get_model(config, mode="semantic"):

    if mode == "panoptic":
        # For panoptic segmentation, create PaPs model
        if config.backbone == "utae":
            from src.panoptic.paps import PaPs

            encoder = UTAE(
                input_dim=10,  # PASTIS has 10 spectral bands
                encoder_widths=eval(config.encoder_widths),
                decoder_widths=eval(config.decoder_widths),
                out_conv=eval(config.out_conv),
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=True,  # Important: set to True for PaPs
                return_maps=True,  # Important: return feature maps
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )

            model = PaPs(
                encoder=encoder,
                num_classes=config.num_classes,
                shape_size=config.shape_size,
                mask_conv=config.mask_conv,
                min_confidence=config.min_confidence,
                min_remain=config.min_remain,
                mask_threshold=config.mask_threshold,
            )
        else:
            raise NotImplementedError(
                f"Backbone {config.backbone} not implemented for panoptic mode"
            )
    elif config.model == "utae":
        model = UTAE(
            input_dim=10,  # Sentinel-2 has 10 bands
            encoder_widths=eval(config.encoder_widths),
            decoder_widths=eval(config.decoder_widths),
            out_conv=eval(config.out_conv),
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
        )
    elif config.model == "uconvlstm":
        model = RecUNet(
            input_dim=10,
            encoder_widths=eval(config.encoder_widths),
            decoder_widths=eval(config.decoder_widths),
            out_conv=eval(config.out_conv),
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            temporal="lstm",
            encoder_norm=config.encoder_norm,
            padding_mode=config.padding_mode,
            pad_value=config.pad_value,
        )
    elif config.model == "buconvlstm":
        model = RecUNet(
            input_dim=10,
            encoder_widths=eval(config.encoder_widths),
            decoder_widths=eval(config.decoder_widths),
            out_conv=eval(config.out_conv),
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            temporal="blstm",
            encoder_norm=config.encoder_norm,
            padding_mode=config.padding_mode,
            pad_value=config.pad_value,
        )
    else:
        raise ValueError(f"Unknown model: {config.model}")

    return model


"""
Get number of trainable parameters
"""


def get_ntrainparams(model):

    return sum(p.numel() for p in model.parameters() if not p.stop_gradient)
