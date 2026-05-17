import math

from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.activations import ACT2FN 
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)  # 升维的维度
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,  # 高频阈值
            "beta_slow": 1,  # 低频阈值
            "factor": 16,  # 扩展倍数，最终的max_position_embeddings = 2048 * factor
            "original_max_position_embeddings": 2048,  # 训练时的最大长度
            "attention_factor": 1.0,  # 注意力缩放因子，默认为1.0，表示不缩放
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter((torch.ones(dim)))

    """
    x shape: [batch_size, seq_len, dim]
    eg: (2, 3, 4) : 一共两句话，每句话三个token(词),每个token的维度是4
    对每个token进行归一化
    最后的输出也是 [batch_size, seq_len, dim]，每个token的维度是4，且每个token的值被归一化了
    """
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
    

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32),
            rope_scaling.get("beta_slow", 1),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 *math.log(rope_base))
            # 计算高频和低频的维度边界
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device) - low).float() / (high - low), 0, 1) # 计算出每个维度的缩放因子
            freqs = freqs * (1 - ramp + ramp / factor)  # 计算出最终每个维度的频率
    t = torch(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  #  freqs[pos, i]=位置 pos在第i对维度上的旋转角度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    q, v :shape = [batch, seq, num_heads, head_dim] 
    批次(同时处理几个句子), 序列长度(每个句子几个词), 注意力头数, 每个头的维度
    cos, sin: shape = [seq, dim(总隐藏维度)]  每个位置的旋转角度,

    最后返回包含位置信息的q_embed, k_embed，shape = [batch, seq, dim]
    每个head使用相同的位置编码
    """
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def reapeat_kv(x: torch.Tensor, n_rep: int):
    """
    x.shape = [bs, slen, num_key_value_heads, head_dim]
    """
    bs, slen, num_key_value_head, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_head, n_rep, head_dim).reshape(bs, slen, num_key_value_head * n_rep, head_dim))


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads  # 总的注意力头数(Q的头数)
        self.n_local_kv_heads = self.num_key_value_heads  # K和 V的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads   # 每个K,V头需要被重复的次数，来匹配Q的头数
        self.head_dim = config.head_dim # 每个头的维度
        self.is_causal = True  # 掩码是否使用
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)  # resNet连接的dropout
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and config.flash_attn


    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        x shape: [batch_size, seq_len, hidden_size]
        position_embeddings: tuple(cos, sin), each shape: [seq_len, head_dim * num_key_value_heads]
        """
        bsz, seq_len, _ = x.shape
        # 投影,计算出Q,K,V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # 每个shape: [batch_size, seq_len, num_heads * head_dim]
        # 把输出的Q,K,V reshape成多头的形式，方便后续计算注意力
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [batch_size, seq_len, num_kv_heads, head_dim]
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  # [batch_size, seq_len, num_kv_heads, head_dim]
        xq, xk = self.q_norm(xq), self.k_norm(xk)  # 对Q和K进行RMSNorm归一化
        # 应用旋转位置编码RoPE，把位置信息融入到Q和K中
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 推理阶段才用到缓存的K,V，训练阶段每次输入一个完整的序列，不需要缓存

        if past_key_value is not None:
            # 如果有缓存的K,V，把它们和当前的K,V拼接起来，形成完整的历史+当前的K,V
            xk = torch.cat([past_key_value[0], xk], dim=1) # [batch_size, total_seq_len, num_kv_heads, head_dim]
            xv = torch.cat([past_key_value[1], xv], dim=1) # [batch_size, total_seq_len, num_kv_heads, head_dim]
        past_kv = (xk, xv) if use_cache else None
        # 对于 K 和 V, 如果 Q 的头数多于 K,V 的头数，则需要重复 K,V 的头来匹配 Q 的头数
        xq, xk, xv = (xq.transpose(1, 2), reapeat_kv(xk, self.n_rep).transpose(1, 2), reapeat_kv(xv, self.n_rep).transpose(1, 2))  # 每个shape: [batch_size, num_heads, seq_len, head_dim]
        # 进行注意力计算，得到每个位置的输出表示
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)  # [batch_size, num_heads, seq_len, head_dim]
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) # [batch_size, num_heads, seq_len, seq_len]
            # 因果掩码, 确保每个位置只能关注之前的位置，不能关注未来的位置
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)  # 添加因果掩码，确保每个位置只能关注之前的位置
            # 注意力掩码, 屏蔽掉padding等不需要关注的位置，确保模型不会把注意力放在这些位置上(处理变长输入)
            if attention_mask is not None:  # attention_mask shape: [batch_size, seq_len], 1表示需要关注的位置，0表示不需要关注的位置
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9  # 添加注意力掩码，屏蔽掉padding等不需要关注的位置, shape: [batch_size, 1, 1, seq_len], 广播机制自动补齐
            output = self.attn_dropout(F.softmax(scores, dim=-1)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)   # [batch_size, seq_len, num_heads * head_dim]
        output = self.resid_dropout(self.o_proj(output))  # 最后通过输出投影，并添加残差连接的dropout
        return output, past_kv
    
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) # 维度: [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, intermediate_size] -> [batch_size, seq_len, hidden_size]
    

class MiniMindBlock(nn.Module):
    def __int__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)  # 多头注意力层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 注意力层的输入归一化
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps) # FNN层的输入归一化
        self.mlp = FeedForward(config) 
    
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value =self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, 
            past_key_value, use_cache, attention_mask
        )
        hidden_states = residual + hidden_states  # 注意力的残差连接
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers  # 词表大小和层数
        # 词嵌入层，把输入的token id shape:[batch, seq_len]  转换成向量表示，shape: [batch, seq_len, hidden_size]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # Transformer层, 每层包含一个多头注意力子层和一个前馈网络子层，层与层之间有残差连接和归一化
        self.layers = nn.ModuleList([MiniMindBlock(i, config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算RoPE的位置编码，存储在模型的buffer中，方便后续使用
        freqs_cos, freqs_sin = precompute_freqs_cis(config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """
        input_ids shape: [batch_size, seq_len]
        attention_mask shape: [batch_size, seq_len], 1表示需要关注的位置，0表示不需要关注的位置
        past_key_values: list of tuples, 每个tuple包含一个注意力层的缓存的K,V，shape: [batch_size, total_seq_len, num_kv_heads, head_dim]
        use_cache: 是否返回新的缓存的K,V
        """
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None   # 兼容 HuggingFace 的缓存格式
        past_key_values = past_key_values or [None] * len(self.layers)  # 如果没有提供缓存的K,V，就创建一个全None的列表，表示每层都没有缓存,  确保循环时每层都有对应的缓存（即使是 None）
        # 计算当前输入是序列的第几个位置，等于之前缓存的K,V的长度，因为每个位置对应一个K,V，所以缓存的长度就是已经处理过的序列长度
        # past_key_values[0]：第0层的缓存 (K, V), past_key_values[0][0]：第0层的 K 缓存, past_key_values[0][0].shape = [batch, seq_len, heads, head_dim], 所有层的 K,V 的 seq_len 都是一样的，等于已经处理过的序列长度
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 计算当前输入的起始位置，等于缓存的K,V的长度
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos : start_pos + seq_length], self.freqs_sin[start_pos : start_pos + seq_length])  # 获取当前输入位置对应的RoPE位置编码
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states, position_embeddings, past_key_value, use_cache, attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)  # 最后再进行一次归一化
        return hidden_states, presents


class MiniMindCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig 
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(config)
        self.model = MiniMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight  # 词嵌入层和输出层共享权重
        self.post_init()  # HuggingFace 的权重初始化方法

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, **kwargs):
        """
        logits_to_keep: int, 在推理阶段，为了节省显存，可以只返回最后几个token的logits，默认为0，表示返回所有token的logits
        """
        hidden_states, past_key_values = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])  # 只计算最后几个token的logits，节省显存
        return CausalLMOutputWithPast(
            logits=logits,  # [batch_size, slice_len, vocab_size] 概率分布，表示每个位置预测下一个token的概率
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )