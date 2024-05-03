import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gaussian_diffusion_utils import _extract_into_tensor, mean_flat


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  (starting at T and going to 1 or reversely?)
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self, *, betas, predict_xstart, learn_sigmas=False,
                 rescale_learned_sigmas, sigma_small,
                 use_kl, rescale_timesteps=False):

        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl
        self.rescale_timesteps = rescale_timesteps

        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        """
            Forward Process: 
            z0 = q(z), z_T ~ N(0, I)
            q(z_t | z_t-1) = N(z_t; sqrt(alpha_t) z_t-1, beta_t I)
            q(z_t | z_0) = N(z_0; sqrt(alpha_t_bar), (1-alpha_t_bar) I)
            
            Precomputed Products: 
            alpha_t_bar = cumprod(alpha_1, ..., alpha_t)
            z_t = sqrt(alpha_t) z_t-1 + sqrt(beta_t) epsilon_t-1
                = sqrt(alpha_t_bar) z_0 + sqrt(1-alpha_t_bar) epsilon

            In diffusion LM: when using sqrt noise schedule:
            alpha_t_bar = 1 - sqrt(t/(T+s)), s as a small constant
        """

        alphas = 1.0 - betas
        # in this case alphas_cumprod[t-1] = alphas_cumprod_prev[t]
        # alphas_cumprod[t+1] = alphas_cumprod_next[t]
        self.alphas_cumprod = np.cumprod(alphas, axis=0)  # alpha_t_bar
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        """
        Reverse Process: recover z0 -- p_theta(z0:T)
        p_theta(z0:T) = p(z_T) PROD_{t=1}^T p_theta(z_t-1|z_t)
        p_theta(z_t-1|z_t) = N(z_t; mu_theta(z_t, t), sigma_theta(z_t, t))

        --> learn f_theta(zt, t), mu_theta() and sigma_theta ~ q(z_t | z_t-1) mean & std
        mu_t(zt, z0) = sqrt(alpha_t)* (1-alpha_t-1_bar)/(1-alpha_t_bar) z_t
                     + sqrt(alpha_t-1_bar) * beta_t / (1-alpha_t_bar) z_0
        """

        # calculations for posterior q(z_{t-1} | z_t, z_0)
        # mu_t(zt, z0): coefficient for z_0 and z_t respectively
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        # var(zt, z0) for q(z_{t-1} | z_t, z_0)
        # log calculation clipped: posterior var is 0 at the beginning of the diffusion chain.
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:]))

        # Not used
        self.mapping_func = None
        self.add_mask_noise = False

    def _predict_xstart_from_eps(self, x_t, t, eps):
        # x_t  = sqrt_alphas_cumprod x_0 + sqrt_one_minus_alphas_cumprod epsilon
        # x_0 = sqrt_recip_alphas_cumprod x_t - sqrt_recipm1_alphas_cumprod epsilon
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        z_t = sqrt(alpha_t_bar) z_0 + sqrt(1-alpha_t_bar) epsilon

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        # z_t = sqrt(alpha_t_bar) z_0 + sqrt(1-alpha_t_bar) epsilon
        x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        p_theta(z_t-1|z_t) = N(z_t; mu_theta(z_t, t), sigma_theta(z_t, t))
        """

        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)

        # sanity check
        assert (posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True,
                        denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0. Adapted from DDPM.
        Returns the predicted mean and variance of p(x_{t-1} | x_t), and predicted x_0.

        :param model: the model, which takes a signal and a batch of timesteps as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # fixed small (p_var = q_posterior_variance) works well for x_0 deterministic
        # fixed large works well for x_0 ~ N(0, I)
        # Here we use fixed large, and set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:
            ### model is used to predict eps
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x, t,
                 clip_denoised=True, denoised_fn=None, model_kwargs=None,
                 top_p=None, mask=None, x_start=None,):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised,
                                   denoised_fn=denoised_fn, model_kwargs=model_kwargs)

        # control the randomness in noise
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = torch.randn_like(x)
            replace_mask = torch.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = torch.randn_like(noise[replace_mask])
                replace_mask = torch.abs(noise) > top_p
            # ensure that the noise values remain within the specified range
            assert (torch.abs(noise) <= top_p).all()

        else:
            noise = torch.randn_like(x)

        # Note here we are using the mean of q_posterior result instead of the
        # one suggested in the original DDPM paper (haven't figured out why)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

        if mask == None:
            pass
        else:
            sample = torch.where(mask==0, x_start, sample)

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"],
            "out": out
        }

    def p_sample_loop(self, model, shape,
                      noise=None, clip_denoised=True, denoised_fn=None,
                      model_kwargs=None, device=None, progress=False,
                      top_p=None, clamp_step=None, clamp_first=None,
                      mask=None, x_start=None, gap=1,):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
                model, shape, noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                top_p=top_p,
                clamp_step=clamp_step,
                clamp_first=clamp_first,
                mask=mask,
                x_start=x_start
        ):
            final.append(sample['sample'])
        return final

    def p_sample_loop_progressive(self, model, shape,
                                  noise=None, clip_denoised=True, denoised_fn=None,
                                  model_kwargs=None, device=None, progress=False,
                                  top_p=None, clamp_step=None, clamp_first=None,
                                  mask=None, x_start=None):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices: # from T to 0
            t = torch.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with torch.no_grad():
                out = self.p_sample(model, sample_x, t,
                                    clip_denoised=clip_denoised,
                                    denoised_fn=denoised_fn_cur,
                                    model_kwargs=model_kwargs,
                                    top_p=top_p, mask=mask, x_start=x_start)
                yield out
                sample_x = out["sample"]

    def _get_x_start(self, x_start_mean, std):
        '''
        Re-parameterization:
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = torch.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return x_start_mean + std * noise

    def _x0_helper(self, model_output, x, t):

        # pred_xprev same as p_mean_var's model mean
        # but we are only using this for loss function

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:  # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev': pred_prev, 'pred_xstart': pred_xstart}

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # batch, seq_len, vocab
        # print(logits.shape)
        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fun(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask != None:
            decoder_nll *= mask
        # print(decoder_nll.shape)
        if mask != None:
            decoder_nll = decoder_nll.sum(dim=-1)/mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll

    def training_losses_seq2seq(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # save the orignal x_0: We are not using this beacause z_0 = noised(x_start_fix)
        x_start_fix = x_start

        assert 'input_ids' in model_kwargs
        input_ids_x = model_kwargs.pop('input_ids').to(t.device)
        input_ids_mask = model_kwargs.pop('input_mask').to(t.device)
        x_start_mean = model.model.module.get_embeds(input_ids_x)   # embed into continuous space

        # Note: std = sqrt(1-alpha_bar)[0] = sqrt(1-alpha)[0] = sqrt(beta_0)
        # This is also why we are using fixedlarge for p_mean_var
        # We are not scaling the mean for the initial step
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
        # x_start_log_var = 2 * torch.log(std)
        x_start = self._get_x_start(x_start_mean, std)      # z_0 term = noised embed(w)

        if noise is None:
            noise = torch.randn_like(x_start)

        # forward t:
        # re-parametrization: z_t = sqrt(alpha_t_bar) z_0 + sqrt(1-alpha_t_bar) epsilon
        x_t = self.q_sample(x_start, t, noise=noise, mask=input_ids_mask)

        """
        simple surrogate objective:
        L_VLB = E[KL(q(zT|z0) || p(zT)) -- L_T
              + sum_{t>1} KL(q(z_t-1,| zt, z0)|| p(x_t-1|xt)) -- L_t-1
              + KL(q(z_0,| w_xy)|| p(z_0|z_1))
              - log p (w_xy | z_0) ]
              
        But here we approximate the ELBO by its lower bound:
        sum_{t=2 to T} ||y0 - f(zt, t)||^2 + ||emb(w_y)-f(z1, 1)|| + R(||z0||)^2
        """

        # model.model.module is for GPU parallelization
        get_logits = model.model.module.get_logits

        terms = {}

        target = x_start  # noised input: emb(w) + std * noise
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)  # x_t noise is masked
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)      # t!=0, y0 and prediction

        # predicted_xstart = model_output if not converted from eps
        # Q: why we need _x0_helper? seems that we can omit this function
        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart']
        t0_mask = (t == 0)
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)    # t=0, w_y and prediction
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])      # mse of shape batch_size

        # tT_mask = (t == self.num_timesteps - 1)
        out_mean, _, _ = self.q_mean_variance(x_start,
                                              torch.LongTensor([self.num_timesteps - 1]).to(x_start.device))
        tT_loss = mean_flat(out_mean ** 2)

        decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x)  # embedding regularization
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask,
                                                 truncate=True, t=t)  # x_0->model_out_x_start
        # assert (model.lm_head.weight == model.word_embedding.weight).all()

        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms


