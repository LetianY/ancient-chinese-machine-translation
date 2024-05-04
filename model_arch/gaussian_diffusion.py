import math
import torch
import numpy as np
from gaussian_diffusion_utils import _extract_into_tensor, mean_flat

import sys
sys.path.append('.')


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

        self.model = None
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

    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

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

    def _token_discrete_loss(self, hidden_x, get_logits, input_ids,
                             mask=None, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0):
        negative log-likelihood (NLL) loss for discrete tokens
        '''
        reshaped_x = hidden_x
        logits = get_logits(reshaped_x)  # batch, seq_len, vocab
        loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fun(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask != None:
            decoder_nll *= mask
        # print(decoder_nll.shape)
        if mask != None:
            # taken mean only at not-masked dims
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
        sum_{t=2 to T} ||y0 - f(zt, t)||^2 + ||emb(w_y)-f(z1, 1)||^2 + R(||z0||)^2
        = mse * C + rounding_loss
        """

        # model.model.module is for GPU parallelization
        get_logits = model.model.module.get_logits

        terms = {}

        target = x_start  # noised input: emb(w) + std * noise
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)  # x_t noise is masked
        assert model_output.shape == target.shape == x_start.shape

        # predicted_xstart = model_output if not converted from eps
        # Q: why we need _x0_helper? seems that we can omit this function
        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart']

        # MSE loss: target = x_start (noised true) -- z0; x_start_mean: w
        terms["mse"] = mean_flat((target - model_out_x_start) ** 2)      # t!=0, y0 and prediction

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

        """
        nll: prediction cross entropy loss
        loss: loss function for diffusion model
        loss = rounding_loss (decoder_nll) + mse_t + alpha_bar * ||x_start||**2 (tT_loss)
        """

        # The choice of C, the choice of rounding loss, and tT_loss might be for stable training
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

    """
    DDIM module directly adapted from original repo to accelerate inference time.
    Need to check the correctness later
    """
    def ddim_sample(self, model, x, t,
                    clip_denoised=True, denoised_fn=None,
                    model_kwargs=None, eta=0.0, langevin_fn=None,
                    mask=None, x_start=None):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(model, x, t,
                                   clip_denoised=clip_denoised,
                                   denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                 * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                     + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1))))  # no noise when t == 0
        # print(sigma.mean())

        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample = langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)

        if mask == None:
            pass
        else:
            sample = torch.where(mask == 0, x_start, sample)

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(self, model, x, t,
                            clip_denoised=True,
                            denoised_fn=None,
                            model_kwargs=None,
                            eta=0.0,):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(model, x, t,
                                   clip_denoised=clip_denoised,
                                   denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = ((_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
               - out["pred_xstart"])
               / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape))
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_next)
                     + torch.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, noise=None,
                         clip_denoised=True, denoised_fn=None,
                         model_kwargs=None, device=None, progress=False,
                         top_p=None, clamp_step=None, clamp_first=None,
                         mask=None, x_start=None, gap=1):
        """
        Generate samples from the model using DDIM.
        :param gap: compute ddim sampling for each {gap} step

        Same usage as p_sample_loop().
        """
        final = []
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                mask=mask,
                x_start=x_start,
                gap=gap
        ):
            final.append(sample['sample'])
        return final

    def ddim_sample_loop_progressive(self, model, shape, noise=None,
                                     clip_denoised=True, denoised_fn=None,
                                     model_kwargs=None, device=None, progress=False,
                                     eta=0.0, langevin_fn=None, mask=None, x_start=None, gap=1):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # print(kwargs.keys())
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called p_mean_var')
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called training_losses')
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)
