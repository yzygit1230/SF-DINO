import torch
import torch.nn as nn
import torch.nn.functional as F



def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class SparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=256, k_size=3, reduction=8, chunk_size=120, conv=default_conv, res_scale=1):
        super(SparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
                                                                                             -1)  
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) 
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  

        hash_codes = torch.argmax(rotated_vecs, dim=-1) 

        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):

        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 64)

        hash_codes = self.LSH(hash_buckets, x_embed)  
        hash_codes = hash_codes.detach()

        _, indices = hash_codes.sort(dim=-1)  
        _, undo_sort = indices.sort(dim=-1)  
        mod_indices = (indices % L) 
        x_embed_sorted = batched_index_select(x_embed, mod_indices)  
        y_embed_sorted = batched_index_select(y_embed, mod_indices)  

        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C)) 
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size, C))  
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)

        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  

        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        ret = torch.reshape(ret, (N, -1, C * self.reduction))  
        bucket_score = torch.reshape(bucket_score, (N, -1,))  
        ret = batched_index_select(ret, undo_sort)  
        bucket_score = bucket_score.gather(1, undo_sort)  

        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret

class FrequencySelectiveHashingAttention(nn.Module):
    def __init__(self,n_hashes=4, embed_dim=128, fft_norm='ortho'):
        # bn_layer not used
        super(FrequencySelectiveHashingAttention, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.sparse_attention = SparseAttention( n_hashes=n_hashes, channels=embed_dim* 2, k_size=3, reduction=4, chunk_size=112, conv=default_conv, res_scale=1)

        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.sparse_attention(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        return output
