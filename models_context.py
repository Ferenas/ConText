from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
import fvcore.nn.weight_init as weight_init
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from util.losses import *
from util.vitdet_utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    LayerNorm2D,
)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, hw_shape=None):
        if len(x.shape) == 4:
            B,H,W,_ = x.shape
            N = H*W
        else:
            B,N,_ = x.shape
            H,W = hw_shape

        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)


        if self.use_rel_pos:
            if len(x.shape) == 4:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            else:
                attn[:,:-1,:-1] = add_decomposed_rel_pos(attn[:,:-1,:-1], q[:,:-1], self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        if len(x.shape) == 4:
            x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        else:
            x = (attn @ v).view(B, self.num_heads, N, -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        
        x = self.proj(x)



        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x, merge=0,hw_shape=None):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x,hw_shape)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # feature ensemble
        if merge > 0:
            prompt, inputs = x.split(x.shape[1] // 2, dim=1)
            if merge == 1:
                num_prompts = x.shape[0] // 3
                inp,inp_o,inp_s = inputs.chunk(3,dim=0)

                inp_1 = torch.cat((inp,inp_o))
                o_shape = inp_1.shape
                inp_1 = inp_1.reshape(2, num_prompts, -1)
                inp_1 = inp_1.mean(dim=1, keepdim=True).expand_as(inp_1).reshape(o_shape)
    

                inp_2 = torch.cat((inp,inp_s))
                o_shape = inp_2.shape
                inp_2 = inp_2.reshape(2, num_prompts, -1)
                inp_2 = inp_2.mean(dim=1, keepdim=True).expand_as(inp_2).reshape(o_shape)


                inputs = torch.cat((0.5*(inp_1[:num_prompts]+inp_2[:num_prompts]),inp_1[num_prompts:],inp_2[num_prompts:]))


            else:
                inputs_mask,inputs_ori = inputs.split(inputs.shape[0] // 2, dim=0)
                inputs_ori = inputs_ori.mean(dim=0, keepdim=True).expand_as(inputs_ori)
                inputs_mask = inputs_mask.mean(dim=0, keepdim=True).expand_as(inputs_mask)
                inputs = torch.cat([inputs_mask, inputs_ori], dim=0)

            x = torch.cat([prompt, inputs], dim=1)
        
        
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.1,
                mlp_ratio=4, act_layer=nn.GELU,drop_path=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm_context = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, query, kv,cat_num=3,switch=False,cot=False):
        
        if len(query.shape) == 4:
            b,h,w,c = kv.shape
            if switch and (cot == False):
                kv = torch.cat((kv[:,-h//2:],kv[:,:h//2]),dim=1)

            if kv.shape[0] != query.shape[0]:
                query,kv = query.reshape(query.shape[0],-1,c),kv.reshape(query.shape[0],cat_num,-1,c).view(query.shape[0],-1,c)
            else:
                query,kv = query.reshape(b,-1,c),kv.reshape(b,-1,c)

        
        shortcut = query
        query = self.norm_context(query)
        kv = self.norm(kv)
        B, N, C = query.shape
        _,Nk,_ = kv.shape
        q = self.wq(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        k = self.wk(kv).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(kv).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)


        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N

        if switch:
            switch_mask = torch.full((N,N),float('-inf')).to(q.device).to(q.dtype)
            split_ind = N // 2
            switch_mask[:N // 2,:N // 2] = 0
            switch_mask[-N // 2:,-N//2:] = 0

            if b != B:
                attn = (attn + switch_mask.reshape(1,1,N,N).repeat(B,self.num_heads,1,cat_num)).softmax(dim=-1)
            else:
                attn = (attn + switch_mask.reshape(1,1,N,N).repeat(B,self.num_heads,1,cat_num)).softmax(dim=-1)

        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C

        #FFN
        x = shortcut.reshape(B,-1,C) + self.drop_path(self.proj_drop(self.proj(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        
        final_x = x.reshape(B,h,w,c)
        
        return final_x




class SegGPT(nn.Module):
    def __init__(
             self,
             img_size=224,
             patch_size=16,
             in_chans=3,
             embed_dim=1024,
             depth=24,
             num_heads=16,
             mlp_ratio=4.,
             qkv_bias=True,
             drop_path_rate=0.,
             norm_layer=nn.LayerNorm,
             act_layer=nn.GELU,
             use_abs_pos=True,
             use_rel_pos=False,
             rel_pos_zero_init=True,
             window_size=0,
             window_block_indexes=(),
             residual_block_indexes=(),
             use_act_checkpoint=False,
             pretrain_img_size=224,
             pretrain_use_cls_token=True,
             out_feature="last_feat",
             decoder_embed_dim=128,
             loss_func="smoothl1",
             ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.ori_h_w = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.reason_mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_r = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        self.context_token = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.context_token_g = nn.Parameter(torch.zeros(1, img_size[1] // patch_size, img_size[1] // patch_size,embed_dim))

        self.new_context_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # token for seg types
        self.type_token_cls = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.type_token_ins = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.type_reson_weight = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        pretrain_patch_size = 16
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // pretrain_patch_size) * (pretrain_img_size // pretrain_patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        self.contextformer = CrossAttention(dim=embed_dim,drop_path=0.3)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()

    
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim*4, patch_size ** 2 * self.decoder_embed_dim, bias=True)  # decoder to patch
        self.decoder_pred = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                LayerNorm2D(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True), # decoder to patch
        )




        self.decoder_mask = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),  #self.decoder_embed_dim
                LayerNorm2D(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, 2, kernel_size=1, bias=True), # decoder to patch
                
        )





        # --------------------------------------------------------------------------
        self.loss_func = loss_func

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.reason_mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        torch.nn.init.normal_(self.segment_token_r, std=.02)


        torch.nn.init.normal_(self.type_token_ins, std=.02)
        torch.nn.init.normal_(self.new_context_token, std=.02)
        torch.nn.init.normal_(self.context_token, std=.02)
        torch.nn.init.normal_(self.type_token_ins, std=.02)
        torch.nn.init.normal_(self.type_reson_weight, std=.02)
        torch.nn.init.normal_(self.context_token_g, std=.02)
        

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % p == 0

        w = imgs.shape[3] // p
        h = w * 2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        w = int((x.shape[1]*0.5)**.5)
        h = w * 2
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward_encoder(self, imgs, tgts, bool_masked_pos, bool_masked_pos_ori, seg_type, merge_between_batch=-1,cot=False):
        # embed patches
        imgs,imgs_ori = imgs[0],imgs[1]
        x = self.patch_embed(imgs)
        y = self.patch_embed(tgts)
        x_ori = self.patch_embed(imgs_ori)

        batch_size, Hp, Wp, C = x.size()
        seq_len = Hp * Wp

        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)
        reason_mask_token = self.reason_mask_token.expand(batch_size, Hp, Wp, -1)
        
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
        y = y * (1 - w) + mask_token * w


        w_ori = bool_masked_pos_ori.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
        x_ori = x_ori*(1-w_ori) + mask_token * (w_ori)




        # add pos embed w/o cls token
        x = x + self.segment_token_x
        x_ori = x_ori + self.segment_token_r
        y = y + self.segment_token_y
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            x_ori = x_ori + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            y = y + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (y.shape[1], y.shape[2])
            )

        # add type tokens for cls and ins
        type_emb = torch.zeros(batch_size, 1, 1, self.type_token_cls.shape[-1]).to(x.device)
        type_emb[seg_type==0] = self.type_token_cls
        type_emb[seg_type==1] = self.type_token_ins

        x = x + type_emb
        x_ori = x_ori + type_emb
        y = y + type_emb

        inter_ind = x.shape[0] 

        x = torch.cat((x,x_ori,y), dim=0)  #2*L (img,tgt), Hp, Wp, dim
                
        merge_idx = 2
        # apply Transformer blocks
        out = []
        
        for idx, blk in enumerate(self.blocks):
            merge = 0
            if merge_between_batch >= 0 and idx >= merge_between_batch:
                merge = 0 if (merge_idx+6) >= idx else 2  #7 for text seg
            x = blk(x, merge=merge,hw_shape=[Hp,Wp])
            if idx == merge_idx: 



                x,x_ori,x_mask = x.chunk(3, dim=0)
                x_fmask = (x + x_mask)*0.5

                x_fori = (x + x_ori)*0.5 





                cat_all = torch.cat((x,x_ori,x_mask),dim=0)



                x_fmask = self.contextformer(x_fmask,cat_all,cat_num=3,switch=True) + x_fmask 
                x_fori =  self.contextformer(x_fori,cat_all,cat_num=3,switch=True) + x_fori


                if merge_between_batch >= 0:
                    x_fmask_p,x_fmask_i = x_fmask.split(x_fmask.shape[1]//2,dim=1)
                    x_fmask_i = x_fmask_i.reshape(x_fmask_i.shape[0],-1)
                    x_fmask_i = x_fmask_i.mean(dim=0,keepdim=True).expand_as(x_fmask_i).reshape(x_fmask_p.shape)
                    x_fmask = torch.cat((x_fmask_p,x_fmask_i),dim=1)


                x = torch.cat([x_fmask,x_fori], dim=0)

            


            if idx in [5, 11, 17, 23]:  
                out.append(self.norm(x))

        return out




    def forward_decoder(self, x):
        x = torch.cat(x, dim=-1)

        x = self.decoder_embed(x) # BxhxwxC
        p = self.patch_size
        h, w = x.shape[1], x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))

        x = self.decoder_pred(x) # Bx3xHxW

        return x


    def forward_decoder_mask(self, x, hr=False,ori=False):
        x = torch.cat(x, dim=-1)
        B = x.shape[0]
        x = self.decoder_embed(x) # BxhxwxC
        
        
        
        p = self.patch_size
        h, w = x.shape[1], x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))


        x_img = self.decoder_pred(x) # Bx3xHxW
        x_g_mask,x_g_ori = x_img[:B//2], x_img[B//2:]


        

        if hr:
            x_mask = self.decoder_mask_hr(x[:B//2] + x_g_mask[:B//2])
        else:
            x_mask = self.decoder_mask(x[:B//2])
        return [x_img,x_mask]



    def denorm(self,tgts):
        imagenet_mean = torch.tensor([[0.485, 0.456, 0.406]]).to(tgts.device).reshape(1,-1,1,1)
        imagenet_std = torch.tensor([[0.229, 0.224, 0.225]]).to(tgts.device).reshape(1,-1,1,1)
        tgts_new = torch.clip((tgts * imagenet_std + imagenet_mean), 0, 1) 
        tgts_new[tgts_new<1] = 0
        tgts_new[tgts_new>0] = 1

        return tgts_new[:,0].to(torch.int64)


    def forward_loss_withmask(self, pred, tgts, mask, mask_re, valid,hr=False):
        """
        tgts: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        valid: [N, 3, H, W]
        """
        mask = mask[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask = self.unpatchify(mask)
        mask = mask * valid


        mask_re = mask_re[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask_re = self.unpatchify(mask_re)
        mask_re = mask_re * valid

        pred,pred_mask = pred
        b = tgts.shape[0]
        target,target_pred = tgts,self.denorm(tgts[:b//2])



        valid_pred_mask = mask[:,0]  
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss_pred = criterion(pred_mask, target_pred)  # (B, H, W)
        masked_loss = loss_pred * valid_pred_mask
        valid_loss = masked_loss.sum() / valid_pred_mask.sum()
        

        
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        
        loss_mask, loss_re = loss.chunk(2,dim=0)    
        loss_re = (loss_re * (mask_re)).sum() / (mask_re).sum()  # mean loss on removed patches
        loss_mask = (loss_mask * mask).sum() / mask.sum()  # mean loss on removed patches
        loss = (loss_mask + loss_re + valid_loss)   

        return loss


    def forward(self, imgs, tgts, bool_masked_pos=None,bool_masked_pos_ori=None,valid=None, seg_type=None, merge_between_batch=-1,val=False,cot=False):
        
        hr = False

        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((imgs.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(imgs.device)
        else:
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

        if bool_masked_pos_ori is None:
            bool_masked_pos_ori = torch.zeros((imgs.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(imgs.device)
        else:
            bool_masked_pos_ori = bool_masked_pos_ori.flatten(1).to(torch.bool)

        
        latent = self.forward_encoder(imgs, tgts, bool_masked_pos, bool_masked_pos_ori, seg_type, merge_between_batch=merge_between_batch,cot=cot)
        pred = self.forward_decoder_mask(latent,hr=hr,ori=imgs[0])
        
    
        new_tgts = torch.cat([tgts,imgs[1]], dim=0)
        if val == False:
            loss = self.forward_loss_withmask(pred, new_tgts, bool_masked_pos,bool_masked_pos_ori,valid,hr=hr)
            return loss, self.patchify(pred[0]), bool_masked_pos
        else:
            return pred[0], bool_masked_pos



def seggpt_vit_large_patch16_input896x448(**kwargs):
    model = SegGPT(
        img_size=(896, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model

def seggpt_vit_large_patch32_input2048x1024(**kwargs):
    model = SegGPT(
        img_size=(2048, 1024), patch_size=32, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)

