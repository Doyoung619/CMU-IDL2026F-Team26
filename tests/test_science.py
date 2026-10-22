import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from joint_control.controller import score,select,armijo,pilot_statistics
from joint_control.model import MLP


class ScientificChecks(unittest.TestCase):
    def test_mlp_gradient_finite_difference(self):
        model=MLP(4,5,3);theta=model.initialize(3)
        rng=np.random.default_rng(9);x=rng.normal(size=(7,4));y=np.arange(7)%3
        _,g=model.loss_gradient(theta,x,y)
        for idx in rng.choice(len(theta),30,replace=False):
            delta=np.zeros_like(theta);delta[idx]=1e-6
            plus=model.loss_gradient(theta+delta,x,y,False)[0]
            minus=model.loss_gradient(theta-delta,x,y,False)[0]
            self.assertAlmostEqual(g[idx],(plus-minus)/2e-6,places=7)

    def test_cross_entropy_and_accuracy(self):
        model=MLP(2,3,2);theta=np.zeros(sum(model.sizes))
        x=np.ones((8,2));y=np.array([0,1]*4)
        metric=model.evaluate(theta,x,y)
        self.assertAlmostEqual(metric['loss'],np.log(2))
        self.assertEqual(metric['accuracy'],.5)

    def test_safe_score_and_budget(self):
        rng=np.random.default_rng(8)
        remaining=3200;horizon=60
        for t in range(horizon):
            g=rng.normal(size=6);v=rng.normal(size=6)
            variance=10**rng.uniform(-2,2)
            batches=[b for b in [16,32,64,128] if b<=remaining-(horizon-t-1)*16]
            b,beta,s=select(g,v,variance,batches,[0.,.5,.9],rng.random())
            self.assertGreaterEqual(s+1e-12,score(g,v,variance,16,0))
            remaining-=b
            self.assertGreaterEqual(remaining,(horizon-t-1)*16)

    def test_stale_momentum_rejected(self):
        g=np.ones(4)
        self.assertEqual(score(g,-g,10,32,.9),0.)
        self.assertGreater(score(g,-g,10,32,0.),0.)

    def test_variance_scaling(self):
        rng=np.random.default_rng(15)
        groups=rng.normal(size=(20000,3))/np.sqrt(8)
        _,variance=pilot_statistics(groups,8)
        self.assertAlmostEqual(variance,3.,delta=.06)

    def test_armijo_brackets_and_monotonicity(self):
        theta=np.array([2.,-3.]);h=np.array([1.,10.]);g=h*theta
        f=lambda z:.5*float(z@(h*z))
        result=armijo(theta,g,g,f,initial=1e-3)
        self.assertEqual(result.status,'bracketed')
        self.assertLessEqual(f(theta-result.eta*g),f(theta)-.5*result.eta*float(g@g))
        self.assertGreater(f(theta-2*result.eta*g),f(theta)-result.eta*float(g@g))
        self.assertEqual(armijo(theta,-g,g,f).eta,0.)

    def test_neural_budget_and_pilot_accounting(self):
        from joint_control.neural import train
        rng=np.random.default_rng(1);x=rng.normal(size=(40,784));y=np.arange(40)%10
        config=dict(hidden=4,gradient_budget=384,pilot_size=64,pilot_groups=4,
                    initial_price=.05,price_step=.01,target_update_batch=64,
                    batch_sizes=[32,64,128],momentums=[0.,.5,.9],log_every=128)
        data=(x,y,x,y,x,y,{})
        for method in ['fixed','cosine','pairwise','joint','fixed_batch','joint_p32']:
            result=train(config,data,method,0,.1,.9)
            self.assertEqual(result['gradient_examples'],384)
            self.assertEqual(sum(r['pilot']+r['batch'] for r in result['steps']),384)
            self.assertTrue(all(np.isfinite(r['eta']) for r in result['steps']))


if __name__=='__main__':
    unittest.main()
