

import torch 
import numpy as np

from torch_utils.ops import filtered_lrelu, bias_act, conv2d_gradfix


def Synth_Input_forward(self, w, t=None):
    # Introduce batch dimension.
    transforms = self.transform  # [batch, row, col]
    freqs = self.freqs.unsqueeze(0)  # [batch, channel, xy]
    phases = self.phases.unsqueeze(0)  # [batch, channel]

    # Apply learned transformation.
    if t is None:
        t = self.affine(w)  # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
        device = w.device
        batch_size = w.shape[0]
    else:
        device = t.device
        batch_size = t.shape[0]

    m_r = torch.eye(3, device=device).unsqueeze(0).repeat([batch_size, 1, 1])  # Inverse rotation wrt. resulting image.
    m_r[:, 0, 0] = t[:, 0]  # r'_c
    m_r[:, 0, 1] = -t[:, 1]  # r'_s
    m_r[:, 1, 0] = t[:, 1]  # r'_s
    m_r[:, 1, 1] = t[:, 0]  # r'_c
    m_t = torch.eye(3, device=device).unsqueeze(0).repeat([batch_size, 1, 1])  # Inverse translation wrt. resulting image.
    m_t[:, 0, 2] = -t[:, 2]  # t'_x
    m_t[:, 1, 2] = -t[:, 3]  # t'_y
    transforms = m_r @ m_t @ transforms  # First rotate resulting image, then translate, and finally apply user-specified transform.

    # Transform frequencies.
    phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
    freqs = freqs @ transforms[:, :2, :2]

    # Dampen out-of-band frequencies that may occur due to the user-specified transform.
    amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)

    # Construct sampling grid.
    theta = torch.eye(2, 3, device=device)
    theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
    theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
    grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

    # Compute Fourier features.
    x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3)  # [batch, height, width, channel]
    x = x + phases.unsqueeze(1).unsqueeze(2)
    x = torch.sin(x * (np.pi * 2))
    x = x * amplitudes.unsqueeze(1).unsqueeze(2)

    # Apply trainable mapping.
    weight = self.weight / np.sqrt(self.channels)
    x = x @ weight.t()

    # Ensure correct shape.
    x = x.permute(0, 3, 1, 2)  # [batch, channel, height, width]
    # misc.assert_shape(x, [batch_size, self.channels, int(self.size[1]), int(self.size[0])])
    return x

def modulated_conv2d(
        x,  # Input tensor: [batch_size, in_channels, in_height, in_width]
        w,  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
        s,  # Style tensor: [batch_size, in_channels]
        demodulate=True,  # Apply weight demodulation?
        padding=0,  # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    # with misc.suppress_tracer_warnings():  # this value will be treated as a constant
    batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    # misc.assert_shape(w, [out_channels, in_channels, kh, kw])  # [OIkk]
    # misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    # misc.assert_shape(s, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0)  # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels)  # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

def Synthesis_Layer_forward(self, x, w, styles=None, noise_mode='random', force_fp32=False, update_emas=False):
    assert noise_mode in ['random', 'const', 'none']  # unused
    #misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])

    # Track input magnitude.
    if update_emas:
        with torch.autograd.profiler.record_function('update_magnitude_ema'):
            magnitude_cur = x.detach().to(torch.float32).square().mean()
            self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
    input_gain = self.magnitude_ema.rsqrt()

    if styles is None:
        #misc.assert_shape(w, [x.shape[0], self.w_dim])
        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

    # Execute modulated conv2d.
    dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
    x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
                            padding=self.conv_kernel - 1, demodulate=(not self.is_torgb), input_gain=input_gain)

    # Execute bias, filtered leaky ReLU, and clamping.
    gain = 1 if self.is_torgb else np.sqrt(2)
    slope = 1 if self.is_torgb else 0.2
    x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
                                        up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

    # Ensure correct shape and dtype.
    #misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
    assert x.dtype == dtype
    return x

def Synthesis_forward(self, ws, all_s=None, **layer_kwargs):

    if all_s is None:
        #misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)

        # Execute layers.
        x = self.input(ws[0])
        for name, w in zip(self.layer_names, ws[1:]):
            x = getattr(self, name)(x, w, **layer_kwargs)
    else:
        t = all_s['input']
        # x = self.input(None, t=t)
        #print("DEBUG 1")
        x = Synth_Input_forward(self.input, None, t=t)
        #print("DEBUG 2")

        for name in self.layer_names:
            # print(name)
            styles = all_s[name]
            # x = getattr(self, name)(x, None, styles=styles, **layer_kwargs)
            x = Synthesis_Layer_forward(getattr(self, name), x, None, styles=styles, **layer_kwargs)
    if self.output_scale != 1:
        x = x * self.output_scale

    # Ensure correct shape and dtype.
    #misc.assert_shape(x, [None, self.img_channels, self.img_resolution, self.img_resolution])
    x = x.to(torch.float32)
    return x

def W2S(self, ws):

    all_s = {}
    ws = ws.to(torch.float32).unbind(dim=1)

    # Execute layers.
    # x = self.input(ws[0])

    t = self.input.affine(ws[0])  # t = (r_c, r_s, t_x, t_y)
    t = t / t[:, :2].norm(dim=1, keepdim=True)  # t' = (r'_c, r'_s, t'_x, t'_y)
    all_s['input'] = t

    for name, w in zip(self.layer_names, ws[1:]):
        layer = getattr(self, name)
        # (x, w, **layer_kwargs)
        styles = layer.affine(w)
        if layer.is_torgb:
            weight_gain = 1 / np.sqrt(layer.in_channels * (layer.conv_kernel ** 2))
            styles = styles * weight_gain
        all_s[name] = styles
    return all_s

def S2Sdict(G, s):
    all_s = {}
    cur_dim = 0
    layer_names = ["input"] + G.synthesis.layer_names
    for name, dim in zip(layer_names, G.SDIMS):
        all_s[name] = s[cur_dim:cur_dim+dim].unsqueeze(0)
        cur_dim += dim
    return all_s

def w2s(G,w):
    all_s  = W2S(G.synthesis, w)
    s = torch.cat([s[0] for s in all_s.values()])
    return s

def s2img(G,s):
    all_s = S2Sdict(G, s)
    img = Synthesis_forward(G.synthesis, None, all_s=all_s)
    return img
