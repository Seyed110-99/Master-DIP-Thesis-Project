import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.optim import Prior

# YOUR PATH TO THE tdv.pt FILE 
TDV_REGULARISER_PATH = "checkpoints/tdv.pt"

def load_tdv_regulariser(device, config=None):
    """
    Load the TDV regularizer with the given configuration.
    """
    if config is None:
        config = {
            "in_channels": 1,
            "num_features": 32,
            "multiplier": 1,
            "num_mb": 3,
            "num_scales": 3,
            "potential": "quadratic",
            "activation": "softplus",
            "zero_mean": True,
        }

    regulariser = TDV(**config).to(device)

    weights = torch.load(
        TDV_REGULARISER_PATH,
        map_location=device,
        weights_only=True,
    )

    prefix_to_remove = "regularizer."
    new_state_dict = {}
    for key, value in weights.items():
        # A. Skip the 'alpha' and 'scale' keys completely
        if key in ["alpha", "scale"]:
            # print(f"Skipping key: {key}")
            continue
        # B. If the key starts with the prefix, remove it
        if key.startswith(prefix_to_remove):
            # Create the new key by stripping the prefix
            new_key = key[len(prefix_to_remove) :]
            new_state_dict[new_key] = value
            # print(f"Remapping '{key}' to '{new_key}'")
        # C. Otherwise, keep the key as is
        else:
            new_state_dict[key] = value
            # print(f"Keeping key as is: {key}")

    regulariser.load_state_dict(new_state_dict)
    return regulariser


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        zero_mean=False,
        bound_norm=False,
        pad=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.reduction_dim = (1, 2, 3)
        self.pad = pad

        p = kernel_size // 2
        self.pad_op = Pad(pad=p, mode="reflect")

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            weight = torch.empty(out_channels, in_channels, 1, 3)
        else:
            weight = torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(weight)
        # insert them using a normal distribution
        nn.init.normal_(self.weight.data, 0.0, math.sqrt(1 / (in_channels * kernel_size**2)))

    def get_weight(self):
        if self.invariant:
            weight = torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                device=self.weight.device,
            )
            weight[:, :, 1, 1] = self.weight[:, :, 0, 0]
            weight[:, :, ::2, ::2] = self.weight[:, :, 0, 2].view(self.out_channels, self.in_channels, 1, 1)
            weight[:, :, 1::2, ::2] = self.weight[:, :, 0, 1].view(self.out_channels, self.in_channels, 1, 1)
            weight[:, :, ::2, 1::2] = self.weight[:, :, 0, 1].view(self.out_channels, self.in_channels, 1, 1)
        else:
            weight = self.weight

        if self.zero_mean:
            weight = weight - torch.mean(weight, dim=self.reduction_dim, keepdim=True)
        if self.bound_norm:
            norm = torch.sum(weight**2, dim=self.reduction_dim, keepdim=True).sqrt()
            weight = weight / norm.clip(min=1)

        return weight

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # compute the convolution
        x = self.pad_op(x)
        return F.conv2d(x, weight, self.bias, self.stride, 0, self.dilation, self.groups)

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[2] - ((x.shape[2] - 1) * self.stride + 1),
                output_shape[3] - ((x.shape[3] - 1) * self.stride + 1),
            )
        else:
            output_padding = 0

        # compute the convolution
        x = F.conv_transpose2d(
            x,
            weight,
            self.bias,
            self.stride,
            0,
            output_padding,
            self.groups,
            self.dilation,
        )
        x = self.pad_op.backward(x)
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class ConvScale2d(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        groups=1,
        stride=2,
        bias=False,
        zero_mean=False,
        bound_norm=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            invariant=invariant,
            stride=stride,
            dilation=1,
            groups=groups,
            bias=bias,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
        )

        # create the convolution kernel
        if self.stride > 1:
            k = torch.as_tensor([1, 4, 6, 4, 1], dtype=torch.float32)[:, None]
            k = k @ k.T
            k /= k.sum()
            self.register_buffer("blur", k.reshape(1, 1, 5, 5))

        # overwrite padding op
        p = (kernel_size + 4 * stride // 2) // 2
        self.pad_op = Pad(pad=p, mode="reflect")

    def get_weight(self):
        weight = super().get_weight()
        if self.stride > 1:
            weight = weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
            for i in range(self.stride // 2):
                weight = F.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(
                self.out_channels,
                self.in_channels,
                self.kernel_size + 2 * self.stride,
                self.kernel_size + 2 * self.stride,
            )
        return weight


class ConvScaleTranspose2d(ConvScale2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        groups=1,
        stride=2,
        bias=False,
        zero_mean=False,
        bound_norm=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            invariant=invariant,
            groups=groups,
            stride=stride,
            bias=bias,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
        )

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)


class Pad(torch.nn.Module):
    def __init__(self, pad, mode="reflect", value=0):
        super().__init__()
        self.pad = pad
        pad = (pad, pad, pad, pad)
        self.f = lambda x: torch.nn.functional.pad(x, pad=pad, mode="constant", value=value)

    def forward(self, x):
        if self.pad > 0:
            x = self.f(x)
        return x

    def backward(self, x):
        if self.pad > 0:
            p = self.pad
            return x[..., p:-p, p:-p]
        else:
            return x


class TDV(Prior):
    """
    total deep variation (TDV) regularizer
    """

    def __init__(
        self,
        in_channels=1,
        num_features=32,
        multiplier=1,
        num_mb=3,
        potential="quadratic",
        activation="softplus",
        zero_mean=True,
        num_scales=3,
    ):
        super().__init__()

        self._fn = self.energy

        self.in_channels = in_channels
        self.num_features = num_features
        self.multiplier = multiplier
        self.num_mb = num_mb
        self.pot, self.act = get_potential(potential)
        self.zero_mean = zero_mean
        self.num_scales = num_scales

        # construct the regularizer
        self.K1 = Conv2d(
            self.in_channels,
            self.num_features,
            3,
            zero_mean=self.zero_mean,
            invariant=False,
            bound_norm=True,
            bias=False,
        )
        self.mb = nn.ModuleList(
            [
                MacroBlock(
                    self.num_features,
                    num_scales=self.num_scales,
                    bound_norm=False,
                    invariant=False,
                    multiplier=self.multiplier,
                    activation=activation,
                )
                for _ in range(self.num_mb)
            ]
        )
        self.KN = Conv2d(self.num_features, 1, 1, invariant=False, bound_norm=False, bias=False)

    def energy(self, x, sigma=1):
        x = self._transformation(x)
        return sigma**2 * self._potential(x)

    def g(self, x, sigma=1):
        return self.energy(x, sigma).sum(dim=(1, 2, 3))

    def grad(self, x, sigma=1, get_energy=False):
        x = self._transformation(x)
        if get_energy:
            energy = self._potential(x).sum(dim=(1, 2, 3)) * sigma**2
        # and its gradient
        x = self._activation(x)
        grad = self._transformation_T(x) * sigma**2
        if get_energy:
            return energy, grad
        else:
            return grad

    def _potential(self, x):
        return self.pot(x) / self.num_features

    def _activation(self, x):
        return self.act(x) / self.num_features

    def _transformation(self, x):
        # extract features
        x = self.K1(x)
        # apply mb
        x = [
            x,
        ] + [None for i in range(self.num_scales - 1)]
        for i in range(self.num_mb):
            x = self.mb[i](x)
        # compute the output
        out = self.KN(x[0])
        return out

    def _transformation_T(self, grad_out):
        # compute the output
        grad_x = self.KN.backward(grad_out)
        # apply mb
        grad_x = [
            grad_x,
        ] + [None for i in range(self.num_scales - 1)]
        for i in range(self.num_mb)[::-1]:
            grad_x = self.mb[i].backward(grad_x)
        # extract features
        grad_x = self.K1.backward(grad_x[0])
        return grad_x


def get_potential(name):
    if name == "linear":
        return lambda x: x, lambda x: torch.ones_like(x)
    elif name == "studentT":
        return lambda x: torch.log(1 + x**2) / 2, lambda x: x / (1 + x**2)
    elif name == "quadratic":
        return lambda x: x**2 / 2, lambda x: x
    elif name == "lncosh":
        return lambda x: -torch.log(1 - torch.tanh(x) ** 2) / 2, torch.tanh
    elif name == "softplus":
        return F.softplus, F.sigmoid
    else:
        raise RuntimeError(f'potential "{name}" not implemented!')


class StudentT_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        d = 1 + x**2
        return torch.log(d) / 2, x / d

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        d = 1 + x**2
        return (x / d) * grad_in1 + (1 - x**2) / d**2 * grad_in2, None


class StudentT2(nn.Module):
    def __init__(self):
        super(StudentT2, self).__init__()

    def forward(self, x):
        return StudentT_fun2.apply(x)


class StudentTgrad_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        d = 1 + (x**2) / 2
        return x / d, (2 - d) / d**2

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        d = 1 + (x**2) / 2
        return ((2 - d) / d**2) * grad_in1 + (x * (d - 4) / d**3) * grad_in2, None


class StudentT2Grad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return StudentTgrad_fun2.apply(x)


class Tanh_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        tanhx = torch.tanh(x)
        ctx.save_for_backward(tanhx)
        return x - tanhx, tanhx**2

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        tanhx = ctx.saved_tensors[0]
        return tanhx**2 * grad_in1 + 2 * (-(tanhx**3) + tanhx) * grad_in2


class Softplus2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Softplus_fun2.apply(x)


@torch.jit.script
def softplusg2(x):
    return torch.exp(x - 2 * F.softplus(x))


class Softplus_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.softplus(x) - math.log(2), F.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        return F.sigmoid(x) * grad_in1 + softplusg2(x) * grad_in2


class Tanh2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Tanh_fun2.apply(x)


def get_activation(name):
    if name == "studentT":
        return StudentT2()
    if name == "studentTgrad":
        return StudentT2Grad()
    elif name == "tanh":
        return Tanh2()
    elif name == "softplus":
        return Softplus2()
    else:
        raise RuntimeError(f'activation "{name}" not implemented!')


class MicroBlock(nn.Module):
    def __init__(self, num_features, bound_norm=False, invariant=False, activation="softplus"):
        super(MicroBlock, self).__init__()

        self.conv1 = Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            invariant=invariant,
            bound_norm=bound_norm,
            bias=False,
        )
        self.act = get_activation(activation)
        self.conv2 = Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            invariant=invariant,
            bound_norm=bound_norm,
            bias=False,
        )

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def forward(self, x):
        a, ap = self.act(self.conv1(x))
        self.act_prime = ap
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        assert not self.act_prime is None
        out = grad_out + self.conv1.backward(self.act_prime * self.conv2.backward(grad_out))
        if not self.act_prime.requires_grad:
            self.act_prime = None
        return out


class MacroBlock(nn.Module):
    def __init__(
        self,
        num_features,
        num_scales=3,
        multiplier=1,
        bound_norm=False,
        invariant=False,
        activation="softplus",
    ):
        super().__init__()

        self.num_scales = num_scales

        # micro blocks
        self.mb = []
        for i in range(num_scales - 1):
            b = nn.ModuleList(
                [
                    MicroBlock(
                        num_features * multiplier**i,
                        bound_norm=bound_norm,
                        invariant=invariant,
                        activation=activation,
                    ),
                    MicroBlock(
                        num_features * multiplier**i,
                        bound_norm=bound_norm,
                        invariant=invariant,
                        activation=activation,
                    ),
                ]
            )
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append(
            nn.ModuleList(
                [
                    MicroBlock(
                        num_features * multiplier ** (num_scales - 1),
                        bound_norm=bound_norm,
                        invariant=invariant,
                        activation=activation,
                    )
                ]
            )
        )
        self.mb = nn.ModuleList(self.mb)

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                ConvScale2d(
                    num_features * multiplier ** (i - 1),
                    num_features * multiplier**i,
                    kernel_size=3,
                    bias=False,
                    invariant=invariant,
                    bound_norm=bound_norm,
                )
            )
            self.conv_up.append(
                ConvScaleTranspose2d(
                    num_features * multiplier ** (i - 1),
                    num_features * multiplier**i,
                    kernel_size=3,
                    bias=False,
                    invariant=invariant,
                    bound_norm=bound_norm,
                )
            )
        self.conv_down = nn.ModuleList(self.conv_down)
        self.conv_up = nn.ModuleList(self.conv_up)

    def forward(self, x):
        assert len(x) == self.num_scales
        # down scale and feature extraction
        for i in range(self.num_scales - 1):
            # 1st micro block of scale
            x[i] = self.mb[i][0](x[i])
            # down sample for the next scale
            x_i_down = self.conv_down[i](x[i])
            if x[i + 1] is None:
                x[i + 1] = x_i_down
            else:
                x[i + 1] = x[i + 1] + x_i_down

        # on the coarsest scale we only have one micro block
        x[self.num_scales - 1] = self.mb[self.num_scales - 1][0](x[self.num_scales - 1])

        # up scale the features
        for i in range(self.num_scales - 1)[::-1]:
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i](x[i + 1], x[i].shape)
            # skip connection
            x[i] = x[i] + x_ip1_up
            # 2nd micro block of scale
            x[i] = self.mb[i][1](x[i])

        return x

    def backward(self, grad_x):
        # backward of up scale the features
        for i in range(self.num_scales - 1):
            # 2nd micro block of scale
            grad_x[i] = self.mb[i][1].backward(grad_x[i])
            # first upsample the next coarsest scale
            grad_x_ip1_up = self.conv_up[i].backward(grad_x[i])
            # skip connection
            if grad_x[i + 1] is None:
                grad_x[i + 1] = grad_x_ip1_up
            else:
                grad_x[i + 1] = grad_x[i + 1] + grad_x_ip1_up

        # on the coarsest scale we only have one micro block
        grad_x[self.num_scales - 1] = self.mb[self.num_scales - 1][0].backward(grad_x[self.num_scales - 1])

        # down scale and feature extraction
        for i in range(self.num_scales - 1)[::-1]:
            # down sample for the next scale
            grad_x_i_down = self.conv_down[i].backward(grad_x[i + 1], grad_x[i].shape)
            grad_x[i] = grad_x[i] + grad_x_i_down
            # 1st micro block of scale
            grad_x[i] = self.mb[i][0].backward(grad_x[i])

        return grad_x
