import h5py
import numpy as np
import pandas as pd
import scipy.sparse

class ChaoticRnn(object):
    def __init__(self, N, p):
        self.N = N
        self.p = p
        self.g = 1.5
        self.alpha = 1.0
        self.dt = 0.1
        self.random_connections()
        self.random_state()

    def random_connections(self):
        mask = np.random.rand(self.N, self.N) < self.p
        scale = 1.0 / np.sqrt(self.p * self.N)
        self.M = np.random.randn(self.N, self.N) * mask * self.g * scale

        self.wo = np.zeros(self.N)
        self.wf = 2.0 * (np.random.rand(self.N) - .5)
        self.dw = np.zeros(self.N)

    def random_state(self):
        self.x0 = 0.5 * np.random.randn(self.N)
        self.z0 = 0.5 * np.random.randn(1)

    def init_state(self):
        self.x = self.x0.copy()
        self.r = np.tanh(self.x)
        self.z = self.z0.copy()

    def train(self, ft, learn_every=1, alpha=1.0):
        # Train the network to replicate the data ft.
        P = np.eye(self.N) / alpha
        outcomes = []
        for t, f in enumerate(ft):
            self.forward()
            if (t + 1) % learn_every == 0:
                # It's learn time!
                k = P.dot(self.r)
                rPr = self.r.dot(k)
                c = 1.0 / (1.0 + rPr)
                outer_product = k.reshape((-1, 1)).dot(k.reshape((1, -1))) * c
                assert outer_product.shape == (self.N, self.N)
                P = P - outer_product
            
                e = self.z - f

                dw = -e * k * c
                self.wo += dw

            outcomes.append(
                {'t': t,
                'z': self.z.copy(),
                'wo_len' : np.linalg.norm(self.wo)
                }
            )
        return pd.DataFrame(outcomes)

    def forward(self):
        self.x = ((1.0 - self.dt) * self.x + 
                  self.dt * self.M.dot(self.r) + 
                  self.dt * self.wf * self.z)
        self.r = np.tanh(self.x)
        self.z = self.wo.dot(self.r)

def _run_tests():
    with h5py.File('inputs.mat', 'r') as f:
        # These are the reference inputs to the Matlab script.
        x0, z0, wf, dt, M, N, ft = (np.array(f['x0/value']).ravel(), 
                                    np.array(f['z0/value']).ravel(), 
                                    np.array(f['wf/value']).ravel(), 
                                    np.array(f['dt/value']).tolist(), 
                                    np.array(f['M/value']).T, 
                                    int(np.array(f['N/value']).tolist()), 
                                    np.array(f['ft/value']).ravel())
        
    with h5py.File('outputs.mat', 'r') as f:
        # These are the outputs
        zt, wo_len, rPrs, es, cs, x_start = np.array(f['zt/value']).ravel(), np.array(f['wo_len/value']).ravel(), np.array(f['rPrs/value']).ravel(), np.array(f['es/value']).ravel(), np.array(f['cs/value']).ravel(), np.array(f['xs/value']).ravel()

    # Instantiate our chaotic RNN.
    rnn = ChaoticRnn(N, 0.1)

    # Make sure that our RNN forward function maintains shape.
    # Force the initial state and connections for testing purposes.
    rnn.x0 = x0.copy()
    rnn.z0 = z0.copy()
    rnn.wf = wf.copy()
    rnn.M = M.copy()
    rnn.init_state()

    xs, rs = rnn.x.shape, rnn.r.shape
    rnn.forward()
    assert xs == rnn.x.shape
    assert rs == rnn.r.shape
    np.testing.assert_allclose(rnn.x, x_start)

    # Force the initial state and connections to be in the initial state again.
    rnn.x0 = x0.copy()
    rnn.z0 = z0.copy()
    rnn.wf = wf.copy()
    rnn.M = M.copy()
    rnn.init_state()
    
    # Run the training.
    results = rnn.train(ft)

    # Test the results
    np.testing.assert_allclose(results['z'].values, zt)
    np.testing.assert_allclose(results['wo_len'], wo_len)

if __name__ == "__main__":
    _run_tests()