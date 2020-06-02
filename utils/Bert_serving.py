# -*- coding: utf-8 -*-
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser


def main():
    args = get_args_parser().parse_args(['-model_dir', r'../data/chinese_L-12_H-768_A-12',
                                         '-port', '86500',
                                         '-port_out', '86501',
                                         '-max_seq_len', '512',
                                         '-mask_cls_sep',
                                         '-cpu'])

    bs = BertServer(args)
    bs.start()


if __name__ == "__main__":
    main()


