import torch


def _passenger_loss(loss, poison_grad, target_grad, target_gnorm):
    """Compute the blind passenger loss term."""
    passenger_loss = 0
    poison_norm = 0

    SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
    if loss == 'top10-similarity':
        _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 10)
    elif loss == 'top20-similarity':
        _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 20)
    elif loss == 'top5-similarity':
        _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 5)
    else:
        indices = torch.arange(len(target_grad))

    for i in indices:
        if loss in ['scalar_product', *SIM_TYPE]:
            passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
        elif loss == 'cosine1':
            passenger_loss -= torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(),
                                                                    dim=0)
        elif loss == 'SE':
            passenger_loss += 0.5 * (target_grad[i] - poison_grad[i]).pow(2).sum()
        elif loss == 'MSE':
            passenger_loss += torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])

        if loss in SIM_TYPE:
            poison_norm += poison_grad[i].pow(2).sum()

    passenger_loss = passenger_loss / target_gnorm  # this is a constant

    if loss in SIM_TYPE:
        passenger_loss = 1 + passenger_loss / poison_norm.sqrt()

    if loss == 'similarity-narrow':
        for i in indices[-2:]:  # normalize norm of classification layer
            passenger_loss += 0.5 * poison_grad[i].pow(2).sum() / target_gnorm

    return passenger_loss


def similarity_loss(loss_type, model, criterion, inputs, labels, target_grad, target_gnorm):
    outputs = model(inputs)
    prediction = (outputs.data.argmax(dim=1) == labels).sum()
    if loss_type in ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']:
        poison_loss = criterion(outputs, labels)
        poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)
        loss = _passenger_loss(loss_type, poison_grad, target_grad, target_gnorm)

    return loss, prediction.detach()