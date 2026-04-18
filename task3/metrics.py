import torch

ACTION_NAMES = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']

def compute_metrics(all_preds, all_targets):
    """
    SR  = fraction of steps where predicted action matches ground truth
          (behavior cloning success rate)
    SPL = SR * path_efficiency
          path_efficiency = correct_steps / total_steps
    NE  = average number of wrong actions (navigation error proxy)
    """
    correct = (all_preds == all_targets).float()
    sr      = correct.mean().item()

    # SPL: penalize by how many wrong steps were taken
    wrong_steps  = (all_preds != all_targets).float().sum().item()
    total_steps  = len(all_preds)
    path_eff     = 1.0 - (wrong_steps / total_steps)
    spl          = sr * max(0, path_eff)

    # Navigation error: average wrong actions per episode (approx)
    ne = wrong_steps / max(1, total_steps / 5)  # assume avg 5 steps/episode

    # Per-action accuracy
    action_acc = {}
    for i, name in enumerate(ACTION_NAMES):
        mask = (all_targets == i)
        if mask.sum() > 0:
            action_acc[name] = (
                all_preds[mask] == all_targets[mask]
            ).float().mean().item()
        else:
            action_acc[name] = 0.0

    return {
        'sr':         round(sr, 4),
        'spl':        round(spl, 4),
        'ne':         round(ne, 4),
        'action_acc': action_acc
    }
