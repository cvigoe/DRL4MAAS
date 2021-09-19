from gym.envs.registration import register

register(
    id='agnosticmaas-v0',
    entry_point='gym_activesearchrlpoisson.envs:AgnosticMAAS',
)

register(
    id='activesearchrlpoisson-v0',
    entry_point='gym_activesearchrlpoisson.envs:ActiveSearchRLPoisson',
)

register(
    id='activesearchrlpoissonmlemap-v0',
    entry_point='gym_activesearchrlpoisson.envs:ActiveSearchRLPoissonMLEMAP',
)

register(
    id='drl4sl-v0',
    entry_point='gym_activesearchrlpoisson.envs:DRL4SL',
)
