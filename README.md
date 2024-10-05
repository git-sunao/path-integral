## Idea/Todo list

- the flow equation does not seem to be holomorphic... This would require modification of jax grad, jacobian part, since it assume holomorphic function.
- normalization of the action, this could matter for the efficiency of flow. For example, the action can be too big when we descretize the path integral using a small time segment, leading the coordinate speed too large.
- Re-using PL thimble for integrand with a similar parrameter family, same idea as done in Feldbrugge. But this time, this can be more efficient, and the criteria of the similarity can be quantified by the KL divergence after learning conditional probability.
- the flow equation is identical for different wave numbers after renormalizing the flow time. This indicates that we do not have to learn proabiblity with different wave number. All we have to do is to modify the integrand.
- The above fact also indicates the efficiency of performing the proper time integral, e.g. in Alice's proper time approach.
- Preprocessing of the MC samples before NF would be nice. -> shift, decorrelate, normalize can be done more efficiently.

## Acknowledgement

- maf is taken from https://github.com/smsharma/jax-conditional-flows
