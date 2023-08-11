import os.path as osp
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

### me 
# _tokenizer = _Tokenizer()
from open_clip import tokenize 
from open_clip.tokenizer import SimpleTokenizer as _Tokenizer
# a = _Tokenizer().encode('man')
_tokenizer = _Tokenizer()
from training.params import parse_args
args = parse_args()
### me 


class PromptLearner(nn.Module):
    def __init__(self, classnames, kg_infos, clip_model, device):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = int ( args.N_CTX )
        dtype = torch.float32 #clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if args.kg_init in ['wikidata', 'wikipedia'] : # KG initialization

            prompts = []
            ctx_vectors = []
        
            for name in classnames:
                ctx_init_for_class = kg_infos[name][args.kg_init]
                
                prompt = tokenize(ctx_init_for_class).to(device=device)  # 1, 77
                
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype) # 1,77,512

                ctx_vectors.append( embedding[0, 1 : 1 + n_ctx, :] )  # n_ctx times [17, 512]
                prompts.append(ctx_init_for_class) # 148
            
            ctx_vectors = torch.stack(ctx_vectors, dim=0)
        
        else: # random initialization
            
            if args.CSC == 'True':
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype).to(device=device)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype).to(device=device)
            
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")
            prompts = [prompt_prefix + " " + name + "." for name in classnames]    

        classnames = [name.replace("_", " ") for name in classnames] #148
        classnames = [name.replace("/", " ") for name in classnames] #148
        name_lens = [len(_tokenizer.encode(name)) for name in classnames] #148 number of tokens(words) per class

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized: 148, 16, 512

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(device=device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # n-cls, n-ctx, d

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts  #148, 77, 512


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float32 #clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x




class COOPCLIP(nn.Module):
    def __init__(self,  classnames, kg_info, clip_model, device):
        super().__init__()
        self.prompt_learner = PromptLearner( classnames, kg_info, clip_model, device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = torch.float32 #clip_model.dtype
        self.device = device

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)) # b, 512

        prompts = self.prompt_learner() #148, 77, 512
        tokenized_prompts = self.tokenized_prompts  # 148, 77
        text_features = self.text_encoder(prompts, tokenized_prompts)  # 148, 512

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # b , 512
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)   # n_class , 512

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logit_scale, logits


