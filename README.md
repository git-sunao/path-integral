# Path-integral
The project of numerical evaluation of path integral. The idea is to utilizing Monte-Carlo (MC) sampling, automatic differentiation (auto diff), and Picard-Lefschetz (PL) theory.

This directory is under development and to be published in a paper later.

## Idea/Todo list

- normalization of the action, this could matter for the efficiency of flow. For example, the action can be too big when we descretize the path integral using a small time segment, leading the coordinate speed too large.
- Re-using PL thimble for integrand with a similar parrameter family, same idea as done in Feldbrugge. But this time, this can be more efficient, and the criteria of the similarity can be quantified by the KL divergence after learning conditional probability.
- the flow equation is identical for different wave numbers after renormalizing the flow time. This indicates that we do not have to learn proabiblity with different wave number. All we have to do is to modify the integrand. The above fact also indicates the efficiency of performing the proper time integral, e.g. in Alice's proper time approach.
- Preprocessing of the MC samples before NF would be nice. -> shift, decorrelate, normalize can be done more efficiently.
- adoptive learning rate? optimizing the architecture? (size of hidden layer, number of transformer)
- optimizing the flow equation
- Aitken algorithm
- Sampler dedicated for Lensing problem: (x-y)^2 + phi. We expect the center of P(x|y) to be y, and the distribution can be captured by Gaussianization after the y-shift.

## DONEs but still need some works
- the flow equation does not seem to be holomorphic... This would require modification of jax grad, jacobian part, since it assume holomorphic function. -> We currently compute the real and imaginary parts of jacobian separately. Is there any better way to speed it up?

## Acknowledgement

- maf is taken from https://github.com/smsharma/jax-conditional-flows
