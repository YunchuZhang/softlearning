        variant = copy.deepcopy(self._variant)

        vae_params = variant['vae_params']
        vae_train_params = variant['vae_train_params']

        vae = ConvVAE(
            vae_params['representation_size'],
            **vae_params['kwargs']
        )

        vae_trainer = ConvVAETrainer(
            vae,
            **vae_train_params['kwargs']
        )

        environment_params = variant['environment_params']
        env_train_params = environment_params['training']
        env_train_params['kwargs']['vae'] = vae

        training_environment = self.training_environment = (
            get_environment_from_params(env_train_params))

        if 'evaluation' in environment_params:
            eval_env_params = environment_params['evaluation']
            eval_env_params['kwargs']['vae'] = vae
            #eval_env_params['kwargs']['render_rollouts'] = True
            #eval_env_params['kwargs']['render_goals'] = True

            evaluation_environment = self.evaluation_environment = (
                get_environment_from_params(eval_env_params))

        else:
            evaluation_environment = training_environment

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        st()
        Qs = self.Qs = get_Q_function_from_variant(
            variant, training_environment)
        policy = self.policy = get_policy_from_variant(
            variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        self.algorithm = get_algorithm_from_variant(
            variant=self._variant,
            VAETrainer=vae_trainer,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=sampler,
            session=self._session)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True