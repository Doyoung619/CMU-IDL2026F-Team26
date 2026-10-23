import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from joint_control.controller import score,select,armijo,pilot_statistics,residual_learning_rate
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

    def test_zero_velocity_momentum_ties_across_scales(self):
        rng=np.random.default_rng(71)
        for scale in [1e-4,1.,1e4]:
            for _ in range(10):
                g=scale*rng.normal(size=17)
                b,beta,s,details=select(g,np.zeros_like(g),100*scale**2,[32],
                                       [.99,.9,.5,0.],0.,return_details=True)
                self.assertEqual(beta,0.)
                self.assertEqual(details['near_tie_count'],4)

    def test_tie_rule_orders_batches_and_respects_tolerance(self):
        from unittest.mock import patch
        values={(32,0.):1.-5e-13,(32,.5):1.,(64,0.):1.-2e-13,(64,.5):1.}
        with patch('joint_control.controller.score',side_effect=lambda g,v,var,b,beta:values[b,beta]):
            b,beta,s,details=select(np.ones(1),np.zeros(1),1.,[64,32],[.5,0.],0.,return_details=True)
            self.assertEqual((b,beta),(32,0.))
            self.assertLessEqual(details['priced_gap'],details['tolerance'])
            self.assertEqual(select(np.ones(1),np.zeros(1),1.,[64,32],[.5,0.],0.,rtol=0,atol=0)[:2],(32,.5))

    def test_residual_never_exceeds_last_accepted_rate(self):
        self.assertEqual(residual_learning_rate(.3,.0625),.0625)
        self.assertEqual(residual_learning_rate(.3,0.),0.)
        self.assertEqual(residual_learning_rate(.3,.5),.3)
        self.assertEqual(residual_learning_rate(.3,.0625,'legacy_fixed'),.3)

    def test_observational_audit_does_not_change_trajectory(self):
        from joint_control.neural import train
        rng=np.random.default_rng(3);x=rng.normal(size=(40,784));y=np.arange(40)%10
        config=dict(hidden=4,gradient_budget=384,pilot_size=64,pilot_groups=4,
                    initial_price=.05,price_step=.01,target_update_batch=64,
                    batch_sizes=[32,64,128],momentums=[0.,.5,.9],log_every=128)
        data=(x,y,x,y,x,y,{})
        control=train(config,data,'joint',0,.1,.9)
        observed=train(dict(config,audit_every=1),data,'joint',0,.1,.9)
        self.assertEqual(control['parameter_sha256'],observed['parameter_sha256'])
        self.assertGreater(len(observed['audits']),0)
        self.assertEqual(observed['diagnostic_gradient_examples'],40*len(observed['audits']))

    def test_tail_keeps_last_accepted_rate_across_a_skipped_step(self):
        from unittest.mock import patch
        from joint_control.neural import train
        from joint_control.controller import SearchResult
        rng=np.random.default_rng(4);x=rng.normal(size=(40,784));y=np.arange(40)%10
        config=dict(hidden=4,gradient_budget=160,pilot_size=32,pilot_groups=4,
                    initial_price=.05,price_step=.01,target_update_batch=32,
                    batch_sizes=[32],momentums=[0.],log_every=160)
        with patch('joint_control.neural.armijo',side_effect=[SearchResult(.0625,2,'bracketed'),SearchResult(0.,0,'non_descent')]):
            result=train(config,(x,y,x,y,x,y,{}),'joint_p32',0,.3,.9)
        tail=result['tail_events'][0]
        self.assertEqual(tail['previous_applied_eta'],0.)
        self.assertEqual(tail['previous_eta'],.0625)
        self.assertEqual(tail['eta'],.0625)

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
