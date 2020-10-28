from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
)


@register_model("position_transformer_align")
class PositionTransformerAlignModel(TransformerModel):
    """
    See "Jointly Learning to Align and Translate with Transformer
    Models" (Garg et al., EMNLP 2019).
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.position_layers = args.position_layers

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(PositionTransformerAlignModel, PositionTransformerAlignModel).add_args(parser)
        parser.add_argument('--position-layers', nargs="+", type=int,
                            help='List of layers where the position attention occurs, starting from 0')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        position_transformer_align(args)

        transformer_model = TransformerModel.build_model(args, task)
        return PositionTransformerAlignModel(
            transformer_model.encoder, transformer_model.decoder, args
        )

    def forward(self, src_tokens, src_lengths, max_target_position,  prev_output_tokens):
        encoder_out, probability = self.forward_encoder(src_tokens, src_lengths, max_target_position)
        return self.decoder(prev_output_tokens, encoder_out), probability

    def forward_encoder(self, src_tokens, src_lengths, max_target_position, **extra_args):
        attn_args = {
            "position_layers": self.position_layers,
        }
        encoder_out, probability = self.encoder(src_tokens, src_lengths, max_target_position, **attn_args, **extra_args)
        return encoder_out, probability


@register_model_architecture("position_transformer_align", "position_transformer_align")
def position_transformer_align(args):
    args.position_layers = getattr(args, "position_layers", [0, 5])
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)


@register_model_architecture("position_transformer_align", "position_transformer_wmt_en_de_big_align")
def position_transformer_wmt_en_de_big_align(args):
    args.position_layers = getattr(args, "position_layers", [0, 5])
    transformer_wmt_en_de_big(args)
