import torch
import torch.nn as nn
from torch.autograd import Variable

# from onmt.modules.UtilClass import BottleLinear
from myS2SVAE.utils.Utils import aeq, sequence_mask


class DotAttention(nn.Module):
    def __init__(self, dim, attn_type="dot", use_cuda = False):
        super(DotAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.attn_output_dim = dim

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): a query `[batch x 1 x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x 1 x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            h_s_ = h_s.transpose(1, 2)
            # (batch, 1, d) x (batch, d, s_len) --> (batch, 1, s_len)
            return torch.bmm(h_t, h_s_)

    def forward(self, input, context, batch_first = True, context_lengths=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x 1 x dim]`
          context (`FloatTensor`): source vectors `[batch x src_len x dim]`
          context_lengths (`LongTensor`): the source context lengths `[batch]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[1 x batch x dim]`
          * Attention distribtutions for each query
             `[1 x batch x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        # Check dimension:
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)
        if context_lengths is not None:
            mask = sequence_mask(context_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            if self.use_cuda == True:
                mask = mask.cuda()
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # # concatenate
        # concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        # attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        # if self.attn_type in ["general", "dot"]:
        #     attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = c.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = c.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        return attn_h, align_vectors

class GeneralAttention(nn.Module):
    def __init__(self, dim, attn_type="general", use_cuda = False):
        super(GeneralAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        self.sm = nn.Softmax()
        self.Wa = nn.Linear(dim, dim, bias=False)
        self.Ua = nn.Linear(2*dim, dim, bias=False)
        self.va = nn.Linear(dim, 1, bias=False)
        self.attn_output_dim = dim * 2

        self.use_cuda = use_cuda and torch.cuda.is_available()

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): a query `[batch x 1 x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x 1 x src_len]`

        """

        # Check input sizes
        # src_batch, src_len, src_dim = h_s.size()
        # tgt_batch, tgt_len, tgt_dim = h_t.size()
        # aeq(src_batch, tgt_batch)
        return self.va(self.Wa(h_t) + self.Ua(h_s)).squeeze()

    def forward(self, input, context, batch_first = True, context_lengths=None):
        """
        Args:
          input (`FloatTensor`): query vectors `[batch x 1 x dim]`
          context (`FloatTensor`): source vectors `[batch x src_len x dim]`
          context_lengths (`LongTensor`): the source context lengths `[batch]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[1 x batch x dim]`
          * Attention distribtutions for each query
             `[1 x batch x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        # Check dimension:
        batch, sourceL, dim = context.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)
        if context_lengths is not None:
            mask = sequence_mask(context_lengths)
            # mask = mask.unsqueeze(1)  # Make it broadcastable.
            if self.use_cuda == True:
                mask = mask.cuda()
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # # concatenate
        # concat_c = torch.cat([c, input], 2).view(batch*targetL, dim*2)
        # attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        # if self.attn_type in ["general", "dot"]:
        #     attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = c.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = c.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        # print(attn_h.size(), align_vectors.size())
        return attn_h, align_vectors


class VaeAttention(nn.Module):
    def __init__(self, z_dim, h_dim, attn_dim, use_cuda = False,attn_method = 'share'):
        super(VaeAttention, self).__init__()
        print(attn_method)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.attn_dim = attn_dim
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.attn_method = attn_method
        self.asign_weight = False

        if attn_method == 'share':
            self.s_attn = torch.nn.Parameter(torch.rand(1), requires_grad=True)
            self.s_z = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        elif attn_method == 'dot':
            self.s_attn = torch.nn.Parameter(torch.rand(h_dim), requires_grad=True)
            self.s_z = torch.nn.Parameter(torch.rand(h_dim), requires_grad=True)
        elif attn_method == 'general':
            self.W_attn = nn.Linear(self.attn_dim, self.h_dim, bias=False)
            self.W_z = nn.Linear(self.z_dim, self.h_dim, bias=False)
            self.H_attn = nn.Linear(self.h_dim, self.h_dim, bias=False)
            self.H_z = nn.Linear(self.h_dim, self.h_dim, bias=False)
            self.v_attn = nn.Linear(self.h_dim, 1, bias=False)
            self.v_z = nn.Linear(self.h_dim, 1, bias=False)
        elif attn_method == 'pointer':
            # sigmoid(h_t * self.w) = attn_score
            self.w = nn.Linear(self.h_dim, 1, bias=False)
        elif attn_method == 'asign_allz' or attn_method == 'asign_alla':
            self.asign_weight = True
            if attn_method == 'asign_allz':
                self.s_z = 1.0
                self.s_attn = 0.0
            elif attn_method == 'asign_alla':
                self.s_attn = 1.0
                self.s_z = 0.0

        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()


    def score(self, attn, z, h):
        if self.attn_method == 'pointer':
            s_attn = nn.Sigmoid()(self.w(h).squeeze(0))
            s_z = 1 - s_attn
            return s_attn, s_z
        elif self.attn_method == 'share':
            s_attn = torch.exp(self.s_attn)
            s_z = torch.exp(self.s_z)
        elif self.attn_method == 'general':
            attn = attn.transpose(0,1)
            z = z.transpose(0,1)
            h = h.transpose(0,1)
            s_attn = torch.sigmoid(torch.bmm(self.W_attn(attn), h.transpose(-1,-2)))
            s_attn = s_attn.transpose(0,1)
            return s_attn, 1-s_attn
        else:
            # s = v^T * (W * a + H * h)
            s_attn = torch.exp(self.v_attn(self.W_attn(attn) + self.H_attn(h)))
            s_z = torch.exp(self.v_z(self.W_z(z) + self.H_z(h)))
        s_plus = s_attn + s_z
        s_attn = s_attn / s_plus
        s_z = s_z / s_plus
        return s_attn, s_z

    def forward(self, attn_vecs, global_vecs, hidden_vecs, batch_first=True, context_lengths=None,
                weight_assign = False, z_w = 1.0, attn_weight = 0.0):

        # Input:
        # attn_vecs: batch_num x str_len x attn_dim
        # global_vecs: batch_num x str_len x z_dim
        # hidden_vecs: batch_num x str_len x h_dim
        # Return:
        # attn: batch_num x str_len x h_dim
        # s_attn: batch_num x strlen x 1
        # s_z: batch_num x strlen x 1
        if self.asign_weight:
            s_attn = self.s_attn
            s_z = self.s_z
            # s_attn = Variable(torch.zeros(1, attn_vecs.size(0), 1), requires_grad=False)
            # s_attn.fill_(attn_weight)
            # s_z = Variable(torch.zeros(1, global_vecs.size(0), 1), requires_grad=False)
            # s_z.fill_(z_w)
            # if self.use_cuda:
            #     s_attn = s_attn.cuda()
            #     s_z = s_z.cuda()
        elif self.attn_method == 'dot':
            # Assuming that attn, z and h already in the same space
            sum = self.s_attn.exp() + self.s_z.exp()
            s_attn = self.s_attn.exp() / sum
            s_z = self.s_z.exp() / sum
        else:
            s_attn, s_z = self.score(attn_vecs, global_vecs, hidden_vecs)

        attn = s_attn * attn_vecs + s_z * global_vecs
        return attn, s_attn, s_z

