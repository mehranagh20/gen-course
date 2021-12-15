HPARAMS_REGISTRY = {}


class Hyperparams(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


abstractart = Hyperparams()
abstractart.width = 384
abstractart.lr = 0.00005
abstractart.latent_dum = 512
abstractart.architecture = "2x2,4m2,4x5,8m4,8x5,16m8,16x10,32m16,32x10,64m32,64x10"
abstractart.dataset = 'abstractart'
abstractart.n_batch = 16
abstractart.image_channels = 3
abstractart.l2_coef = 1.0
abstractart.lpips_coef = 0.05
abstractart.image_size = 64
HPARAMS_REGISTRY['abstractart'] = abstractart


def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)


def add_vae_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--data_root2', type=str, default='./')

    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--restore_ema_path', type=str, default=None)
    parser.add_argument('--restore_log_path', type=str, default=None)
    parser.add_argument('--restore_optimizer_path', type=str, default=None)
    parser.add_argument('--restore_latent_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')

    parser.add_argument('--ema_rate', type=float, default=0.999)

    parser.add_argument('--enc_blocks', type=str, default=None)
    parser.add_argument('--architecture', type=str, default=None)
    parser.add_argument('--zdim', type=int, default=16)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--custom_width_str', type=str, default='')
    parser.add_argument('--bottleneck_multiple', type=float, default=0.25)

    parser.add_argument('--no_bias_above', type=int, default=64)
    parser.add_argument('--scale_encblock', action="store_true")

    parser.add_argument('--test_eval', action="store_true")
    parser.add_argument('--warmup_iters', type=float, default=0)

    parser.add_argument('--num_mixtures', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=200.0)
    parser.add_argument('--skip_threshold', type=float, default=400.0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--lr_prior', type=float, default=0.00015)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--wd_prior', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.9)

    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--iters_per_ckpt', type=int, default=25000)
    parser.add_argument('--iters_per_print', type=int, default=1000)
    parser.add_argument('--iters_per_save', type=int, default=10000)
    parser.add_argument('--iters_per_images', type=int, default=10000)
    parser.add_argument('--epochs_per_eval', type=int, default=10)
    parser.add_argument('--epochs_per_probe', type=int, default=None)
    parser.add_argument('--epochs_per_eval_save', type=int, default=20)
    parser.add_argument('--num_images_visualize', type=int, default=8)
    parser.add_argument('--num_variables_visualize', type=int, default=6)
    parser.add_argument('--num_temperatures_visualize', type=int, default=3)
    parser.add_argument('--num_comp_indices', type=int, default=2)
    parser.add_argument('--num_simp_indices', type=int, default=7)
    parser.add_argument('--dci_num_levels', type=int, default=2)
    parser.add_argument('--dci_field_of_view', type=int, default=10)
    parser.add_argument('--dci_prop_to_retrieve', type=float, default=0.002)
    parser.add_argument('--imle_db_size', type=int, default=1024)
    parser.add_argument('--imle_factor', type=float, default=1.)
    parser.add_argument('--imle_staleness', type=int, default=7)
    parser.add_argument('--imle_batch', type=int, default=128)
    parser.add_argument('--n_overfit', type=int, default=128)
    parser.add_argument('--n_split', type=int, default=8192)
    parser.add_argument('--min_res_for_loss', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--normalize_grad', type=int, default=1)
    parser.add_argument('--lpips_loss', type=int, default=1)
    parser.add_argument('--imle_perturb_coef', type=float, default=0.1)
    parser.add_argument('--lpips_net', type=str, default='vgg')
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--subset_len', type=int, default=-1)
    parser.add_argument('--load_latents', type=int, default=0)
    parser.add_argument('--reinitialize_nn', type=int, default=0)
    parser.add_argument('--proj_dim', type=int, default=1000)
    parser.add_argument('--proj_proportion', type=int, default=0)
    parser.add_argument('--lpips_coef', type=float, default=1.0)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    parser.add_argument('--force_factor', type=int, default=1.5)
    parser.add_argument('--change_threshold', type=float, default=0.17)
    parser.add_argument('--change_coef', type=float, default=0.04)
    parser.add_argument('--n_mpl', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--fname', type=str, default='testing.png')
    return parser
