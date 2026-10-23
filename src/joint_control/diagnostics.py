"""Training-only observational audits; these never control optimizer decisions."""
import numpy as np


def full_loss_gradient(model, theta, x, y, chunk=512):
    loss=0.; gradient=np.zeros_like(theta)
    for start in range(0,len(y),chunk):
        labels=y[start:start+chunk]
        part, grad=model.loss_gradient(theta,x[start:start+chunk],labels)
        weight=len(labels)/len(y)
        loss+=weight*part; gradient+=weight*grad
    return float(loss),gradient


def classify_update(before, after, alignment, gradient_norm, direction_norm, eta, status):
    loss_tol=1e-12*(1+abs(before))
    alignment_tol=1e-12*max(1.,gradient_norm*direction_norm)
    return dict(accepted_loss_increase=bool(eta>0 and after>before+loss_tol),
                skipped_full_descent=bool(status=='non_descent' and alignment>alignment_tol),
                loss_tolerance=loss_tol,alignment_tolerance=alignment_tol)


def audit_update(model,theta,direction,pilot_gradient,x,y,eta,status):
    before,g=full_loss_gradient(model,theta,x,y)
    after=model.evaluate(theta-eta*direction,x,y)['loss'] if eta>0 else before
    alignment=float(g@direction)
    result=dict(full_training_loss_before=before,full_training_loss_after=after,
                full_training_alignment=alignment,pilot_alignment=float(pilot_gradient@direction),
                eta=eta,status=status,gradient_examples=len(y),
                forward_examples=len(y) if eta>0 else 0)
    result.update(classify_update(before,after,alignment,float(np.linalg.norm(g)),
                                  float(np.linalg.norm(direction)),eta,status))
    return result
