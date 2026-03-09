import argparse
import sys
import os

# Ensure the project root is on sys.path so absolute imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from examples.train_aloha_sim_na import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    # parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--num_initial_traj_collect', default=1, help='number of trajectories to collect before starting online updates', type=int)
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)

    parser.add_argument('--action_chunk_size', default=-1, help='Action chunk size', type=int)
    parser.add_argument('--max_timesteps', default=-1, help='Max timesteps', type=int)
    parser.add_argument('--success_threshold', default=1.2, help='Success threshold for considering episode successful', type=float)
    parser.add_argument('--restore_path', default='', help='Path to restore model from', type=str)
    parser.add_argument('--use_local_policy', default=1, help='Use local policy (1) or websocket remote policy (0)', type=int)
    parser.add_argument('--save_kv_cache', default=1, help='Save kv_cache during trajectory collection (1) or not (0)', type=int)
    parser.add_argument('--grl_noise_sample', default=0, help='For noise critic distillation, sample half the noise from the noise actor', type=int)
    parser.add_argument('--flow_integration_steps', default=10, help='Number of flow Euler integration steps for Coop NA policy', type=int)
    parser.add_argument('--online_buffer_size', default=2000, help='Size of the online replay buffer', type=int)
    parser.add_argument('--action_critic_steps', default=30, help='Number of gradient steps for action critic per training step', type=int)
    parser.add_argument('--noise_critic_steps', default=10, help='Number of gradient steps for noise critic per training step', type=int)
    parser.add_argument('--noise_actor_steps', default=20, help='Number of gradient steps for noise actor per training step', type=int)
    parser.add_argument('--train_all_together', default=0, help='Whether to train action critic, noise critic, and noise actor together', type=int)
    parser.add_argument('--put_kv_cache_on_cpu', default=0, help='Whether to put kv_cache on CPU to save GPU memory. This overwrites the kv_cache_device setting', type=int)
    parser.add_argument('--noise_scale_inside', default=0, help='If 1, noise scaling is done inside the tanh distribution (dsrl repo behaviour). If 0, scaling is done externally after sampling (qam repo behavior).', type=int)

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        backup_entropy=False,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    assert variant.action_chunk_size != -1
    assert variant.max_timesteps != -1
    variant['use_local_policy'] = variant.use_local_policy == 1
    variant['save_kv_cache'] = variant.save_kv_cache == 1
    variant['grl_noise_sample'] = variant.grl_noise_sample == 1
    variant['train_all_together'] = variant.train_all_together == 1
    variant['put_kv_cache_on_cpu'] = variant.put_kv_cache_on_cpu == 1
    variant['noise_scale_inside'] = variant.noise_scale_inside == 1
    main(variant)
    sys.exit()
    