
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class DPGroupNorm(nn.Module):
    # GroupNorm replacement for LayerNorm (required for Opacus)
    def __init__(self, hidden_size, num_groups=32, eps=1e-12):
        super(DPGroupNorm, self).__init__()
        if hidden_size % num_groups != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_groups ({num_groups})")
        
        self.num_groups = num_groups
        self.eps = eps
        # GroupNorm expects (batch, channels, *) but input is (batch, seq, hidden)
        # Manually reshape in forward
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x):
        # x shape: (batch, seq, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape to (batch, hidden_size, seq)
        x = x.transpose(1, 2)
        
        # Apply group normalization
        # GroupNorm normalizes over (H/G, W) where G is num_groups
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        
        # Reshape back to (batch, seq, hidden_size)
        x = x.transpose(1, 2)
        
        # Apply affine transformation
        x = x * self.weight + self.bias
        
        return x

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # Use reshape instead of view for Opacus compatibility
        x = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight)
        x = x.reshape(size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0
        # Use reshape instead of view for Opacus compatibility
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).reshape(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        # Use reshape instead of view for Opacus compatibility
        return x.reshape(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # Use reshape instead of view for Opacus compatibility
        x = x.reshape(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        # CHANGE: Use DPGroupNorm instead of LayerNorm
        self.ln_1 = DPGroupNorm(nx, num_groups=32, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = DPGroupNorm(nx, num_groups=32, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class CoarseTransformerModel(nn.Module):
    def __init__(self, config):
        super(CoarseTransformerModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        # CHANGE: Use DPGroupNorm instead of LayerNorm
        self.ln_f = DPGroupNorm(config.n_embd, num_groups=32, eps=config.layer_norm_epsilon)

    def forward(self, input_visits, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_visits.size(1) + past_length, dtype=torch.long,
                                        device=input_visits.device)
            position_ids = position_ids.unsqueeze(0).expand(input_visits.size(0), input_visits.size(1))

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for block, layer_past in zip(self.h, past):
            hidden_states, _ = block(hidden_states, layer_past)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class AutoregressiveLinear(nn.Linear):
    """
    Autoregressive linear layer with triangular mask.
    DP-compatible version without in-place operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, input):
        # Compute triangular mask on-the-fly
        if not hasattr(self, '_mask') or self._mask.device != self.weight.device:
            self._mask = torch.tril(torch.ones(self.out_features, self.in_features, 
                                               device=self.weight.device, 
                                               dtype=self.weight.dtype))
        return F.linear(input, self._mask * self.weight, self.bias)

class FineAutoregressiveHead(nn.Module):
    def __init__(self, config):
        super(FineAutoregressiveHead, self).__init__()
        self.auto1 = AutoregressiveLinear(config.n_embd * 2, config.n_embd * 2)
        self.auto2 = AutoregressiveLinear(config.n_embd * 2, config.total_vocab_size * 2)
        self.n_embd = config.n_embd
        self.tot_vocab = config.total_vocab_size
        self.code_vocab = config.code_vocab_size
        # CHANGE: Store relu as module (not in-place)
        self.relu = nn.ReLU()  # No inplace=True for DP compatibility

    def forward(self, history, input_visits):
        history = history[:,:-1,:]
        input_visits = input_visits[:,1:,:]
        concat = torch.cat((history, input_visits), dim=2)
        # CHANGE: Use module relu (not in-place)
        auto1_out = self.relu(self.auto1(concat))
        auto2_out = self.auto2(auto1_out)
        code_logits = auto2_out[:,:,:self.code_vocab]
        return code_logits

    def sample(self, history, input_visits):
        history = history[:,:-1,:]
        input_visits = input_visits[:,1:,:]
        currVisit = torch.cat((history, input_visits), dim=2)[:,-1,:].unsqueeze(1)
        # CHANGE: Use module relu (not in-place)
        auto1_out = self.relu(self.auto1(currVisit))
        auto2_out = self.auto2(auto1_out)
        code_logits = auto2_out[:,:,:self.code_vocab]
        return code_logits

class DPHALOModel(nn.Module):
    """
    Differential Privacy compatible HALO model.
    
    Key modifications for Opacus:
    1. LayerNorm → DPGroupNorm
    2. No in-place operations
    3. All modules support per-example gradients
    """
    def __init__(self, config):
        super(DPHALOModel, self).__init__()
        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)
        self.config = config

    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None, pos_loss_weight=None):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, hidden_states)
        
        if ehr_labels is not None:
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            shift_labels = shift_labels[:,:,:self.ehr_head.code_vocab]
            
            loss_weights = None
            if pos_loss_weight is not None:
                loss_weights = torch.ones(code_logits.shape, device=code_logits.device)
                loss_weights = loss_weights + (pos_loss_weight-1) * shift_labels
            if ehr_masks is not None:
                code_logits_masked = code_logits * ehr_masks
                shift_labels = shift_labels * ehr_masks
                if pos_loss_weight is not None:
                    loss_weights = loss_weights * ehr_masks
            else:
                code_logits_masked = code_logits

            bce_with_logits = nn.BCEWithLogitsLoss(weight=loss_weights, reduction='none')
            elementwise_loss = bce_with_logits(code_logits_masked, shift_labels)
            
            if ehr_masks is not None:
                valid_elements = ehr_masks.sum()
                loss = elementwise_loss.sum() / (valid_elements + 1e-8)
            else:
                loss = elementwise_loss.mean()
            
            sig = nn.Sigmoid()
            code_probs = sig(code_logits)
            return loss, code_probs, shift_labels
        
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        return code_probs

    def sample(self, input_visits, random=True):
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits)
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, hidden_states)
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)
            
            remaining_visit = visit[:,0,i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:,-1,i + first_nonzero] = visit[:,0,i + first_nonzero]
            i = i + first_nonzero + 1
        
        return input_visits


def validate_dp_compatibility(model):
    """
    Validate that model is compatible with Opacus.
    
    Returns:
        bool: True if compatible, False otherwise
        list: List of compatibility issues (empty if compatible)
    """
    try:
        from opacus.validators import ModuleValidator
        
        errors = ModuleValidator.validate(model, strict=False)
        
        if errors:
            print("\nModel has DP compatibility issues:")
            for error in errors:
                print(f"   - {error}")
            print("\nTry running: model = ModuleValidator.fix(model)")
            return False, errors
        else:
            print("\nModel is DP-compatible.")
            return True, []
            
    except ImportError:
        print("\nOpacus not installed. Install with: pip install opacus")
        return False, ["Opacus not installed"]


if __name__ == "__main__":
    # Demo: Create DP-compatible model and validate
    print("\nDEMONSTRATION: DP-Compatible HALO Model\n")
    
    # Create minimal config
    class TestConfig:
        n_layer = 2
        n_embd = 768
        n_head = 12
        n_ctx = 100
        n_positions = 100
        total_vocab_size = 1000
        code_vocab_size = 900
        attn_pdrop = 0.1
        resid_pdrop = 0.1
        layer_norm_epsilon = 1e-5
    
    config = TestConfig()
    
    # Create DP model
    print("Creating DP-compatible HALO model...")
    model = DPHALOModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 10
    input_visits = torch.randn(batch_size, seq_len, config.total_vocab_size)
    
    try:
        output = model(input_visits)
        print(f"  Forward pass successful.")
        print(f"    Output shape: {output.shape}")
    except Exception as e:
        print(f"  Forward pass failed: {e}")
    
    # Validate DP compatibility
    print("\nValidating DP compatibility...")
    is_compatible, issues = validate_dp_compatibility(model)
    
    if is_compatible:
        print("\nModel is ready for DP training with Opacus.")
    else:
        print("\nModel needs fixes before DP training.")
    
    print("\nDemonstration complete.")
