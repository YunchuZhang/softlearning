#!/bin/sh

softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_car1 --replay_pool=HerReplayPool --algorithm=SAC --mesh=car1
softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_car2 --replay_pool=HerReplayPool --algorithm=SAC --mesh=car2
softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_car3 --replay_pool=HerReplayPool --algorithm=SAC --mesh=car3
softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_car4 --replay_pool=HerReplayPool --algorithm=SAC --mesh=car4
softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_car5 --replay_pool=HerReplayPool --algorithm=SAC --mesh=car5
#softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_can1 --replay_pool=HerReplayPool --algorithm=SAC --mesh=can1
#softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_mug1 --replay_pool=HerReplayPool --algorithm=SAC --mesh=mug1
#softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_mug2 --replay_pool=HerReplayPool --algorithm=SAC --mesh=mug2
#softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_mug3 --replay_pool=HerReplayPool --algorithm=SAC --mesh=mug3
#softlearning run_example_debug examples.development --universe=gym --checkpoint-frequency=50 --domain=SawyerPushAndReachEnvEasy --task=v0 --trial-gpus=1 --exp-name=exp_1_printer --replay_pool=HerReplayPool --algorithm=SAC --mesh=printer
