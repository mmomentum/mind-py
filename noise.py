import itertools
import numpy as nd

try:
    from pyfftw.interfaces.numpy_fft import irfft  # Performs much better than numpy's fftpack
except ImportError:  # Use monkey-patching np.fft perhaps instead?
    from numpy.fft import irfft  # pylint: disable=ungrouped-imports


# 1 dimensional normalization (for audio files)
def normalize(y):
    m = max(abs(y))
    n_factor = 1 / m
    return y * n_factor


def noise(N, color='white', state=None):
    try:
        return _noise_generators[color](N, state)
    except KeyError:
        raise ValueError("Incorrect color.")


def white(N, state=None):
    state = nd.random.RandomState() if state is None else state
    return state.randn(N)


def pink(N, state=None):
    state = nd.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = nd.sqrt(nd.arange(len(X)) + 1.)  # +1 to avoid divide by zero
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def blue(N, state=None):
    state = nd.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = nd.sqrt(nd.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def brown(N, state=None):
    state = nd.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (nd.arange(len(X)) + 1)  # Filter
    y = (irfft(X / S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


def violet(N, state=None):
    state = nd.random.RandomState() if state is None else state
    uneven = N % 2
    X = state.randn(N // 2 + 1 + uneven) + 1j * state.randn(N // 2 + 1 + uneven)
    S = (nd.arange(len(X)))  # Filter
    y = (irfft(X * S)).real
    if uneven:
        y = y[:-1]
    return normalize(y)


_noise_generators = {
    'white': white,
    'pink': pink,
    'blue': blue,
    'brown': brown,
    'violet': violet,
}


def noise_generator(N=44100, color='white', state=None):
    #yield from itertools.cycle(noise(N, color)) # Python 3.3
    for sample in itertools.cycle(noise(N, color, state)):
        yield sample

