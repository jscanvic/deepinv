import torch
import deepinv as dinv

device = "cuda:0"

x = dinv.utils.load_example("butterfly.png")

def _periodic_interp_kernel(t: torch.Tensor, N: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Periodic Dirichlet interpolation kernel on a length-N grid:
      h(t) = (1/N)*sin(pi t)/sin(pi t/N)                 if N odd
           = (1/N)*sin(pi t)/sin(pi t/N)*cos(pi t/N)     if N even
    with t understood modulo N (period N).
    """
    # bring t to [0, N)
    t_mod = torch.remainder(t, N)
    num = torch.sin(torch.pi * t_mod)
    den = torch.sin(torch.pi * t_mod / N).clamp_min(eps)
    h = (num / den) / N
    if (N % 2) == 0:
        h = h * torch.cos(torch.pi * t_mod / N)
    # removable singularity at multiples of N -> limit is 1
    h = torch.where(t_mod == 0, torch.ones_like(h), h)
    return h


def periodic_fractional_shift(
    x: torch.Tensor,
    disp: tuple[float, float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Shift an image tensor x (B,C,H,W) by a fractional displacement (dy, dx) in pixels,
    using the periodic Dirichlet kernel (circular, band-limited interpolation).

    Mathematically:
      y[i, j] = sum_l sum_m x[l, m] * h(i + dy - l) * h(j + dx - m)
    where h is the kernel above, with N=H for rows and N=W for cols.

    Args:
        x:  (B,C,H,W) tensor
        disp: (dy, dx) floats, positive dy shifts content *down*, dx shifts *right*
        eps: small constant for numerical stability in the kernel

    Returns:
        (B,C,H,W) shifted tensor with wrap-around boundaries.
    """
    if x.ndim != 4:
        raise ValueError("x must be (B,C,H,W)")

    B, C, H, W = x.shape
    dy, dx = disp
    device, dtype = x.device, x.dtype

    # Build row kernel matrix Wy[i, l] = h(i + dy - l), size (H, H)
    i = torch.arange(H, device=device, dtype=dtype).view(-1, 1)       # (H,1) output row idx
    l = torch.arange(H, device=device, dtype=dtype).view(1, -1)       # (1,H) input row idx
    Wy = _periodic_interp_kernel(i + dy - l, H, eps=eps)              # (H,H)

    # Build col kernel matrix Wx[o, j] = h(o + dx - j), size (W, W)
    o = torch.arange(W, device=device, dtype=dtype).view(-1, 1)       # (W,1) output col idx
    j = torch.arange(W, device=device, dtype=dtype).view(1, -1)       # (1,W) input col idx
    Wx = _periodic_interp_kernel(o + dx - j, W, eps=eps)              # (W,W)

    # Apply separably: first along H, then along W
    # out1[b,c,i,w] = sum_l Wy[i,l] * x[b,c,l,w]
    out1 = torch.einsum('il,bclw->bciw', Wy, x)
    # out[b,c,i,o] = sum_j Wx[o,j] * out1[b,c,i,j]
    out = torch.einsum('oj,bcij->bcio', Wx, out1)
    return out.contiguous()


# --- Optional: FFT-based version (same result, faster on large H,W) ---
def periodic_fractional_shift_fft(
    x: torch.Tensor,
    disp: tuple[float, float],
) -> torch.Tensor:
    """
    Same shift using Fourier phase ramps (still circular / band-limited).
    """
    if x.ndim != 4:
        raise ValueError("x must be (B,C,H,W)")
    dy, dx = disp
    B, C, H, W = x.shape
    device = x.device

    # Frequency grids with FFT ordering
    ky = torch.fft.fftfreq(H, d=1.0).to(device)  # shape (H,)
    kx = torch.fft.fftfreq(W, d=1.0).to(device)  # shape (W,)
    # Phase: exp(-i 2π (ky*dy + kx*dx))
    phase_y = torch.exp(-2j * torch.pi * ky.view(H, 1) * dy)  # (H,1)
    phase_x = torch.exp(-2j * torch.pi * kx.view(1, W) * dx)  # (1,W)
    phase = phase_y * phase_x                                  # (H,W)

    Xf = torch.fft.fft2(x, dim=(-2, -1))
    Yf = Xf * phase.view(1, 1, H, W)
    y = torch.fft.ifft2(Yf, dim=(-2, -1))
    print(y.imag.abs().pow(2).sum())
    y = y.real
    return y

y = periodic_fractional_shift_fft(x, disp=(0, 128.5))
y = y.clamp(0, 1)

print(x.min(), x.max())
print(y.min(), y.max())

dinv.utils.plot([ x, y ])
