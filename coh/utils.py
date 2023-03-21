import torch


def prepend_bos_token(model, input_ids: torch.LongTensor):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if model.config.is_encoder_decoder:
        input_ids = model._shift_right(input_ids)
    else:
        bs, _ = input_ids.shape
        bos = torch.LongTensor([[model.config.bos_token_id]]).repeat(bs, 1).to(input_ids.device)
        input_ids = torch.cat([bos, input_ids[:, :-1]], dim=1)
    return input_ids
