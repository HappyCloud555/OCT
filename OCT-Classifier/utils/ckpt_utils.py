""" The Code is under Tencent Youtu Public Rule
ckpt related utils
"""
import os
import torch


def save_checkpoint(state, path, logger, is_val, filename='checkpoint.pth.tar'):
    filepath = os.path.join(path, filename)
    if is_val:
        logger.info(f" =====================val best save (details below) best_acc:{state['best_acc']:.2f}=====================")
        torch.save(state, filepath)
    if not is_val:
        logger.info(f" =====================test best save (details below) best_acc:{state['best_acc']:.2f}=====================")
        torch.save(state, os.path.join(path, "model_best.pth.tar"))


def save_ckpt_dict(args, model, ema_model, epoch, batch, optimizer,
                   scheduler, best_acc, logger, is_val=False):
    model_to_save = model.module if hasattr(model, "module") else model
    if args.use_ema:
        ema_to_save = ema_model.ema.module if hasattr(
            ema_model.ema, "module") else ema_model.ema
    to_save_ckpt_dict = {
        'past_epoch': epoch,
        'past_batch': batch,
        'state_dict': model_to_save.state_dict(),
        'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    save_checkpoint(to_save_ckpt_dict, args.out, logger, is_val)


def get_test_model(ema_model, model, use_ema):
    """ use ema model or test model
    """
    if use_ema:
        test_model = ema_model.ema
        test_prefix = "ema"
    else:
        test_model = model
        test_prefix = ""
    return test_model, test_prefix