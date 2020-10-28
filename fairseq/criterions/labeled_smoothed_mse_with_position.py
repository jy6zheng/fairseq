import math

from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion("label_smoothed_mse_with_position")
class LabelSmoothedMSEWithPosition(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, alignment_lambda):
        super().__init__(task, sentence_avg, label_smoothing)
        self.alignment_lambda = alignment_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--alignment-lambda",
            default=0.05,
            type=float,
            metavar="D",
            help="weight for the alignment loss",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, probability = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        alignment_loss = None

        # Compute alignment loss only for training set and non dummy batches.
        if "alignments" in sample and sample["alignments"] is not None:
            alignment_loss = self.compute_alignment_loss(sample, net_output, probability)

        if alignment_loss is not None:
            logging_output["alignment_loss"] = utils.item(alignment_loss.data)
            loss += self.alignment_lambda * alignment_loss

        return loss, sample_size, logging_output

    def compute_alignment_loss(self, sample, net_output, probability):
        total_loss = 0
        for layer in probability:
            # B x T x P -> B x P x T
            layer = layer.transpose(1, 2)
            bsz, tgt_sz, src_sz = layer.shape
            prob = layer.reshape(bsz * tgt_sz, src_sz)
            align = sample["alignments"]
            if len(align) > 0:
                loss = (
                    ((prob[align[:, 1][:, None], align[:, 0][:, None]] - 1)**2)
                ).sum()*(1/len(align))
                total_loss += loss
            else:
                return None
        return total_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        alignment_loss_sum = utils.item(
            sum(log.get("alignment_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "alignment_loss",
            alignment_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True