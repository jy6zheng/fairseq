import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from .cross_entropy import CrossEntropyCriterion


def compute_alignment_loss(sample, probability):
    align = sample["alignments"]
    target = F.one_hot(align[:, :, -1])
    num_layers, num_sentences, src_len, max_target_position = probability.size()
    all_target = torch.empty(num_layers, num_sentences, src_len, max_target_position)
    for i in range(num_layers):
        all_target[i] = target
    word_mse = ((probability - all_target) ** 2).mean(axis=-1)
    alignment_loss = word_mse.sum
    return alignment_loss


@register_criterion("cross_entropy_position")
class CrossEntropyPosition(CrossEntropyCriterion):
    def __init__(self, task, sentence_avg, align_lambda):
        super().__init__(self, task, sentence_avg)
        self.align_lambda = align_lambda

    @staticmethod
    def add_args(parser):
        CrossEntropyPosition.add_args(parser)
        parser.add_arguement("--align-lambda",
                             default=0.6,
                             type=float,
                             metavar="D",
                             help="weight for alignment loss that will be divided over number of layers")

    def forward(self, model, sample, reduce=True):
        net_output, probability = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        alignment_loss = None

        # Calculate alignment loss for training set
        if "alignments" in sample and sample["alignments"] is not None:
            alignment_loss = compute_alignment_loss(sample, probability)

        if alignment_loss is not None:
            logging_output["alignment_loss"] = utils.item(alignment_loss.data)
            loss += self.align_lambda * alignment_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        alignment_loss_sum = sum(log.get("alignment_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "alignment_loss", alignment_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True



